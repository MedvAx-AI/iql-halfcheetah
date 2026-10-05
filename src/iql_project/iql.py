"""Implicit Q-Learning agent: expectile V, twin-Q backup and weighted cloning.

Update order, matching Kostrikov, Nair and Levine (ICLR 2022):

1. Fit V to the upper expectile of ``min(target Q1, target Q2)`` on dataset
   actions. Target critics are stop-gradients (frozen copies, no grad).
2. Fit both online critics to
   ``y = reward + discount * (1 - terminated) * V(next)``.
   Truncation is not part of this mask. ``V(next)`` is detached, so the backup
   does not differentiate through the value network. There is no target V and
   no maximization over actions.
3. Advantage-weighted behavior cloning. Advantage is
   ``min(target Q) - V`` after the value step, detached. Weights are
   ``min(exp(inverse_temperature * advantage), max_weight)``, computed in a
   log-stable form. The policy term is ``-weight * log π(a|s)`` for the dataset
   action; gradients flow only through the squashed-Gaussian likelihood.
4. Soft-update each target critic toward its online critic:
   ``θ' ← (1 - target_tau) θ' + target_tau θ``.

``q_loss`` is ``MSE(Q1, y) + MSE(Q2, y)``, so each critic receives the gradient
of its own mean squared error. ``v_loss`` and ``policy_loss`` are batch means.
``advantage_mean`` is the batch mean of the detached advantage used for the
policy weights.
"""

import copy
import math
import platform
import random
import subprocess
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from . import __version__
from .config import ProjectConfig
from .contracts import DatasetInfo, FloatArray, Metrics, TransitionBatch
from .networks import (
    HIDDEN_DIM,
    HIDDEN_LAYERS,
    LOG_STD_MAX,
    LOG_STD_MIN,
    QNetwork,
    SquashedGaussianPolicy,
    ValueNetwork,
    build_networks,
)

CHECKPOINT_SCHEMA_VERSION = 1
_ARCHITECTURE = {
    "hidden_dim": HIDDEN_DIM,
    "hidden_layers": HIDDEN_LAYERS,
    "log_std_min": LOG_STD_MIN,
    "log_std_max": LOG_STD_MAX,
    "policy": "tanh_squashed_diagonal_gaussian",
    "target_q": "min_of_frozen_twin_critics",
}


def expectile_loss(residual: torch.Tensor, expectile: float) -> torch.Tensor:
    """Mean of ``|τ - 1(u < 0)| u²`` with ``u = Q - V`` and upper expectile ``τ``."""
    weight = torch.where(residual < 0, 1.0 - expectile, expectile)
    return (weight * residual.square()).mean()


def bellman_target(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    """Bootstrap unless the transition is a true termination. Truncation is ignored."""
    bootstrap = (~terminated).to(dtype=next_values.dtype)
    return rewards + discount * bootstrap * next_values


def advantage_weights(
    advantage: torch.Tensor, inverse_temperature: float, max_weight: float
) -> torch.Tensor:
    """``min(exp(β A), max_weight)`` without overflowing the exponential."""
    capped_logit = torch.clamp(inverse_temperature * advantage, max=math.log(max_weight))
    return capped_logit.exp()


def twin_q_loss(q1: torch.Tensor, q2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum of the two critics' mean squared errors against the same detached target."""
    return torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)


def weighted_behavior_cloning_loss(log_prob: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``E[-stopgrad(w) log π(a|s)]``. ``weights`` is detached inside this function."""
    return -(weights.detach() * log_prob).mean()


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak average. ``target`` parameters are updated in place and must not require grad."""
    with torch.no_grad():
        pairs = zip(target.parameters(), source.parameters(), strict=True)
        for target_param, source_param in pairs:
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config.device is 'cuda' but CUDA is not available")
    return torch.device(name)


def source_commit() -> str:
    candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for root in candidates:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
    return "unknown"


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "iql_project": __version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def _python_float(value: torch.Tensor) -> float:
    number = float(value.detach().cpu())
    if not math.isfinite(number):
        raise RuntimeError(f"Non-finite IQL metric: {number}")
    return number


def _frozen_array(values: np.ndarray) -> np.ndarray:
    array = np.array(values, dtype=np.float32, copy=True)
    array.setflags(write=False)
    return array


def dataset_info_from_checkpoint(path: str | Path) -> DatasetInfo:
    """Rebuild ``DatasetInfo`` from a checkpoint without constructing networks.

    Evaluation should call this, construct ``IQLAgent(config, info)``, then
    ``load``. Raw observations are normalized with the returned mean and std
    before ``act``; rewards reported to the user stay in environment units.
    """
    return _dataset_info_from_payload(_read_checkpoint(path))


def config_from_checkpoint(path: str | Path) -> ProjectConfig:
    """Return the exact config stored in a checkpoint."""
    payload = _read_checkpoint(path)
    return ProjectConfig(**payload["config"])


def _read_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    _validate_payload(payload)
    return payload


def _dataset_info_from_payload(payload: dict[str, Any]) -> DatasetInfo:
    preprocessing = payload["preprocessing"]
    return DatasetInfo(
        dataset_id=str(payload["dataset_id"]),
        observation_dim=int(preprocessing["observation_dim"]),
        action_dim=int(preprocessing["action_dim"]),
        action_low=_frozen_array(preprocessing["action_low"]),
        action_high=_frozen_array(preprocessing["action_high"]),
        observation_mean=_frozen_array(preprocessing["observation_mean"]),
        observation_std=_frozen_array(preprocessing["observation_std"]),
        reward_scale=float(preprocessing["reward_scale"]),
        reward_shift=float(preprocessing["reward_shift"]),
    )


def _validate_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema: {payload.get('schema_version')!r}")
    if payload.get("architecture") != _ARCHITECTURE:
        raise ValueError("Checkpoint architecture does not match this implementation")


def _preprocessing_record(info: DatasetInfo) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for field in fields(info):
        value = getattr(info, field.name)
        record[field.name] = value.tolist() if isinstance(value, np.ndarray) else value
    return record


class IQLAgent:
    def __init__(self, config: ProjectConfig, info: DatasetInfo) -> None:
        if info.dataset_id != config.dataset_id:
            raise ValueError(
                f"Dataset ID mismatch: config has {config.dataset_id}, info has {info.dataset_id}"
            )
        self.config = config
        self.info = info
        self.device = resolve_device(config.device)
        networks = build_networks(config, info)
        self.q1: QNetwork = networks["q1"].to(self.device)
        self.q2: QNetwork = networks["q2"].to(self.device)
        self.value: ValueNetwork = networks["value"].to(self.device)
        self.policy: SquashedGaussianPolicy = networks["policy"].to(self.device)
        self.target_q1: QNetwork = networks["target_q1"].to(self.device)
        self.target_q2: QNetwork = networks["target_q2"].to(self.device)
        self.q_optimizer = Adam(
            [*self.q1.parameters(), *self.q2.parameters()], lr=config.learning_rate
        )
        self.v_optimizer = Adam(self.value.parameters(), lr=config.learning_rate)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.learning_rate)
        self._step = 0
        self.elapsed_seconds = 0.0
        self._sampler: np.random.Generator | None = None
        self._sampler_state: dict[str, Any] | None = None

    @property
    def step(self) -> int:
        return self._step

    def bind_sampler(self, rng: np.random.Generator) -> None:
        """Remember the dataset sampler so ``save`` can resume the same batches."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        self._sampler = rng

    def update(self, batch: TransitionBatch) -> Metrics:
        """One sequential IQL update. Returns finite Python floats."""
        observations, actions, rewards, next_observations, terminated = self._batch_tensors(batch)
        self.q1.train()
        self.q2.train()
        self.value.train()
        self.policy.train()

        self._zero_grads()
        v_loss = self._value_loss(observations, actions)
        v_loss.backward()
        self._assert_grads_isolated(self.value)
        self.v_optimizer.step()

        self._zero_grads()
        q_loss = self._q_loss(observations, actions, rewards, next_observations, terminated)
        q_loss.backward()
        self._assert_grads_isolated(self.q1, self.q2)
        self.q_optimizer.step()

        self._zero_grads()
        policy_loss, advantage_mean = self._policy_loss(observations, actions)
        policy_loss.backward()
        self._assert_grads_isolated(self.policy)
        self.policy_optimizer.step()

        metrics = {
            "q_loss": _python_float(q_loss),
            "v_loss": _python_float(v_loss),
            "policy_loss": _python_float(policy_loss),
            "advantage_mean": _python_float(advantage_mean),
        }
        soft_update(self.target_q1, self.q1, self.config.target_tau)
        soft_update(self.target_q2, self.q2, self.config.target_tau)
        self._step += 1
        return metrics

    def act(self, observation: FloatArray, *, deterministic: bool = True) -> FloatArray:
        """Map one normalized observation ``(17,)`` to a bounded action ``(6,)``."""
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != (self.info.observation_dim,):
            raise ValueError(
                f"observation must have shape {(self.info.observation_dim,)}, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("observation contains non-finite values")
        self.policy.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.array(array, copy=True), device=self.device).unsqueeze(0)
            action = self.policy(tensor, deterministic=deterministic)
        result = np.asarray(action.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
        low = np.asarray(self.info.action_low, dtype=np.float32)
        high = np.asarray(self.info.action_high, dtype=np.float32)
        if result.shape != low.shape or np.any(result < low) or np.any(result > high):
            raise RuntimeError("Policy produced an action outside the dataset bounds")
        return result

    def save(self, path: Path) -> None:
        """Save networks, optimizers, step, RNGs, config, preprocessing and versions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rng_state: dict[str, Any] = {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "sampler": copy.deepcopy(self._sampler.bit_generator.state) if self._sampler else None,
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "architecture": dict(_ARCHITECTURE),
            "step": self._step,
            "elapsed_seconds": self.elapsed_seconds,
            "config": asdict(self.config),
            "dataset_id": self.info.dataset_id,
            "preprocessing": _preprocessing_record(self.info),
            "versions": runtime_versions(),
            "git_commit": source_commit(),
            "networks": {name: module.state_dict() for name, module in self._modules().items()},
            "optimizers": {
                "q": self.q_optimizer.state_dict(),
                "v": self.v_optimizer.state_dict(),
                "policy": self.policy_optimizer.state_dict(),
            },
            "rng": rng_state,
        }
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        """Restore a checkpoint into an agent built for the same dataset contract."""
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        _validate_payload(payload)
        if payload["dataset_id"] != self.info.dataset_id:
            raise ValueError(
                f"Checkpoint dataset {payload['dataset_id']} does not match {self.info.dataset_id}"
            )
        self._require_same_preprocessing(_dataset_info_from_payload(payload))
        modules = self._modules()
        for name, module in modules.items():
            module.load_state_dict(payload["networks"][name])
        self.target_q1.requires_grad_(False).eval()
        self.target_q2.requires_grad_(False).eval()
        self.q_optimizer.load_state_dict(payload["optimizers"]["q"])
        self.v_optimizer.load_state_dict(payload["optimizers"]["v"])
        self.policy_optimizer.load_state_dict(payload["optimizers"]["policy"])
        self._move_optimizer_state(self.q_optimizer)
        self._move_optimizer_state(self.v_optimizer)
        self._move_optimizer_state(self.policy_optimizer)
        self._step = int(payload["step"])
        self.elapsed_seconds = float(payload["elapsed_seconds"])
        rng_state = payload["rng"]
        torch.set_rng_state(rng_state["torch"].cpu())
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        if "torch_cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        self._sampler_state = copy.deepcopy(rng_state.get("sampler"))
        if self._sampler is not None and self._sampler_state is not None:
            self._restore_bound_sampler()

    def _value_loss(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_q = torch.minimum(
                self.target_q1(observations, actions), self.target_q2(observations, actions)
            )
        return expectile_loss(target_q - self.value(observations), self.config.expectile)

    def _q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_v = self.value(next_observations)
            target = bellman_target(rewards, next_v, terminated, self.config.discount)
        return twin_q_loss(self.q1(observations, actions), self.q2(observations, actions), target)

    def _policy_loss(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target_q = torch.minimum(
                self.target_q1(observations, actions), self.target_q2(observations, actions)
            )
            advantage = target_q - self.value(observations)
            weights = advantage_weights(
                advantage, self.config.inverse_temperature, self.config.max_weight
            ).squeeze(-1)
        log_prob = self.policy.log_prob(observations, actions)
        return weighted_behavior_cloning_loss(log_prob, weights), advantage.mean()

    def _batch_tensors(
        self, batch: TransitionBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = self._tensor(batch.observations, np.float32, (self.info.observation_dim,))
        actions = self._tensor(batch.actions, np.float32, (self.info.action_dim,))
        rewards = self._tensor(batch.rewards, np.float32, (1,))
        next_observations = self._tensor(
            batch.next_observations, np.float32, (self.info.observation_dim,)
        )
        terminated = self._tensor(batch.terminated, np.bool_, (1,))
        truncated = self._tensor(batch.truncated, np.bool_, (1,))
        count = observations.shape[0]
        tensors = (observations, actions, rewards, next_observations, terminated, truncated)
        if count == 0 or any(tensor.shape[0] != count for tensor in tensors):
            raise ValueError("TransitionBatch fields must share a positive batch dimension")
        if not all(torch.isfinite(tensor).all() for tensor in tensors[:4]):
            raise ValueError("TransitionBatch contains non-finite values")
        low = torch.as_tensor(
            np.array(self.info.action_low, dtype=np.float32, copy=True), device=self.device
        )
        high = torch.as_tensor(
            np.array(self.info.action_high, dtype=np.float32, copy=True), device=self.device
        )
        if torch.any(actions < low) or torch.any(actions > high):
            raise ValueError("Batch actions fall outside the dataset bounds")
        return observations, actions, rewards, next_observations, terminated

    def _tensor(
        self, value: np.ndarray, dtype: np.dtype, trailing: tuple[int, ...]
    ) -> torch.Tensor:
        array = np.asarray(value)
        expected = np.dtype(dtype)
        if array.ndim != 2 or array.shape[1:] != trailing or array.dtype != expected:
            raise ValueError(
                f"Expected dtype {expected} and trailing shape {trailing}, "
                f"got {array.dtype} {array.shape}"
            )
        return torch.as_tensor(np.array(array, copy=True), device=self.device)

    def _zero_grads(self) -> None:
        self.q_optimizer.zero_grad(set_to_none=True)
        self.v_optimizer.zero_grad(set_to_none=True)
        self.policy_optimizer.zero_grad(set_to_none=True)

    def _assert_grads_isolated(self, *allowed: nn.Module) -> None:
        allowed_ids = {id(module) for module in allowed}
        leaked = [
            name
            for name, module in self._modules().items()
            if id(module) not in allowed_ids
            and any(param.grad is not None for param in module.parameters())
        ]
        if leaked:
            raise RuntimeError(f"Gradient leaked into {', '.join(leaked)}")

    def _modules(self) -> dict[str, nn.Module]:
        return {
            "q1": self.q1,
            "q2": self.q2,
            "value": self.value,
            "policy": self.policy,
            "target_q1": self.target_q1,
            "target_q2": self.target_q2,
        }

    def _move_optimizer_state(self, optimizer: Adam) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _require_same_preprocessing(self, saved: DatasetInfo) -> None:
        if (saved.observation_dim, saved.action_dim) != (
            self.info.observation_dim,
            self.info.action_dim,
        ):
            raise ValueError("Checkpoint dimensions do not match the agent")
        pairs = (
            (saved.action_low, self.info.action_low),
            (saved.action_high, self.info.action_high),
            (saved.observation_mean, self.info.observation_mean),
            (saved.observation_std, self.info.observation_std),
        )
        if any(not np.array_equal(left, right) for left, right in pairs):
            raise ValueError("Checkpoint preprocessing does not match the agent DatasetInfo")
        if (saved.reward_scale, saved.reward_shift) != (
            self.info.reward_scale,
            self.info.reward_shift,
        ):
            raise ValueError("Checkpoint reward transform does not match the agent DatasetInfo")

    def _restore_bound_sampler(self) -> None:
        assert self._sampler is not None
        assert self._sampler_state is not None
        if type(self._sampler.bit_generator).__name__ != self._sampler_state.get("bit_generator"):
            raise ValueError("Checkpoint sampler RNG is not a NumPy PCG64 Generator")
        self._sampler.bit_generator.state = copy.deepcopy(self._sampler_state)
