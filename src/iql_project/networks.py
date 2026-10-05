"""Twin Q, state-value network and tanh-squashed Gaussian policy.

Architecture is the standard MuJoCo IQL width: two ReLU hidden layers of 256.
It is intentionally not a ``ProjectConfig`` field; changing the shared schema
needs a lead review. Action bounds come from ``DatasetInfo`` and stay in
environment units.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .config import ProjectConfig
from .contracts import DatasetInfo

HIDDEN_DIM = 256
HIDDEN_LAYERS = 2
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
_SQUASH_EPS = 1e-6


def _mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(HIDDEN_LAYERS):
        layers.extend((nn.Linear(width, HIDDEN_DIM), nn.ReLU()))
        width = HIDDEN_DIM
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Scalar critic Q(s, a). The twin is two independent modules, not a shared trunk."""

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = _mlp(observation_dim + action_dim, 1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((observations, actions), dim=-1))


class ValueNetwork(nn.Module):
    """Scalar state value V(s)."""

    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.net = _mlp(observation_dim, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class SquashedGaussianPolicy(nn.Module):
    """Diagonal Gaussian in pre-tanh space, mapped affinely onto action bounds.

    Likelihood of an environment action ``a``:

    ``u = 2 (a - low) / (high - low) - 1``
    ``z = atanh(u)``
    ``log π(a|s) = log N(z; μ(s), σ(s)) - log(1 - u²) - log((high - low) / 2)``

    summed over action dimensions. The last term is the affine Jacobian; it is
    zero when bounds are ``[-1, 1]``. Dataset actions that sit on the bound are
    clamped by ``1e-6`` before ``atanh`` so the density stays finite.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        super().__init__()
        low = np.asarray(action_low, dtype=np.float32)
        high = np.asarray(action_high, dtype=np.float32)
        if low.shape != (action_dim,) or high.shape != (action_dim,):
            raise ValueError("Action bounds must have shape (action_dim,)")
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
            raise ValueError("Action bounds must be finite and low < high")
        trunk: list[nn.Module] = []
        width = observation_dim
        for _ in range(HIDDEN_LAYERS):
            trunk.extend((nn.Linear(width, HIDDEN_DIM), nn.ReLU()))
            width = HIDDEN_DIM
        self.trunk = nn.Sequential(*trunk)
        self.mean_head = nn.Linear(HIDDEN_DIM, action_dim)
        self.log_std_head = nn.Linear(HIDDEN_DIM, action_dim)
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -0.5)
        self.register_buffer("action_low", torch.tensor(low))
        self.register_buffer("action_high", torch.tensor(high))

    def distribution_params(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observations)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def squash(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        unit = torch.tanh(pre_tanh)
        scale = (self.action_high - self.action_low) / 2
        return self.action_low + (unit + 1) * scale

    def forward(self, observations: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        mean, log_std = self.distribution_params(observations)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        return self.squash(pre_tanh)

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.distribution_params(observations)
        scale = (self.action_high - self.action_low) / 2
        unit = (actions - self.action_low) / scale - 1
        unit = unit.clamp(-1 + _SQUASH_EPS, 1 - _SQUASH_EPS)
        pre_tanh = torch.atanh(unit)
        log_normal = Normal(mean, log_std.exp()).log_prob(pre_tanh)
        log_det_tanh = torch.log(1 - unit.square() + _SQUASH_EPS)
        log_det_scale = torch.log(scale.clamp_min(_SQUASH_EPS))
        return (log_normal - log_det_tanh - log_det_scale).sum(dim=-1)


def build_networks(config: ProjectConfig, info: DatasetInfo) -> dict[str, nn.Module]:
    """Return ``q1``, ``q2``, ``value``, ``policy``, ``target_q1`` and ``target_q2``.

    Target critics are frozen copies. No optimizer is created here.
    """
    if (info.observation_dim, info.action_dim) != (config.observation_dim, config.action_dim):
        raise ValueError("DatasetInfo dimensions do not match the project config")
    q1 = QNetwork(info.observation_dim, info.action_dim)
    q2 = QNetwork(info.observation_dim, info.action_dim)
    target_q1 = copy.deepcopy(q1).requires_grad_(False).eval()
    target_q2 = copy.deepcopy(q2).requires_grad_(False).eval()
    return {
        "q1": q1,
        "q2": q2,
        "value": ValueNetwork(info.observation_dim),
        "policy": SquashedGaussianPolicy(
            info.observation_dim, info.action_dim, info.action_low, info.action_high
        ),
        "target_q1": target_q1,
        "target_q2": target_q2,
    }
