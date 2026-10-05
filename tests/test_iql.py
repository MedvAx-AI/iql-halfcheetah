"""Person 3: analytical IQL losses, masks, bounds, checkpoints and resume."""

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Normal

from iql_project.config import ProjectConfig
from iql_project.contracts import DatasetInfo, TransitionBatch
from iql_project.dataset import ArrayOfflineDataset
from iql_project.iql import (
    IQLAgent,
    advantage_weights,
    bellman_target,
    config_from_checkpoint,
    dataset_info_from_checkpoint,
    expectile_loss,
    soft_update,
    twin_q_loss,
    weighted_behavior_cloning_loss,
)
from iql_project.networks import HIDDEN_DIM, build_networks
from iql_project.train import train

ROOT = Path(__file__).resolve().parents[1]


def make_info(
    *,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> DatasetInfo:
    low = (
        np.full(6, -1, dtype=np.float32)
        if action_low is None
        else np.asarray(action_low, dtype=np.float32)
    )
    high = (
        np.full(6, 1, dtype=np.float32)
        if action_high is None
        else np.asarray(action_high, dtype=np.float32)
    )
    return DatasetInfo(
        dataset_id="mujoco/halfcheetah/medium-v0",
        observation_dim=17,
        action_dim=6,
        action_low=low,
        action_high=high,
        observation_mean=np.zeros(17, dtype=np.float32),
        observation_std=np.ones(17, dtype=np.float32),
        reward_scale=1.0,
        reward_shift=0.0,
    )


def make_config(**overrides: object) -> ProjectConfig:
    return replace(ProjectConfig(), batch_size=8, **overrides)


def make_agent(seed: int = 0, **overrides: object) -> IQLAgent:
    torch.manual_seed(seed)
    return IQLAgent(make_config(**overrides), make_info())


def make_batch(
    count: int = 4,
    *,
    seed: int = 0,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
    info: DatasetInfo | None = None,
) -> TransitionBatch:
    info = make_info() if info is None else info
    rng = np.random.default_rng(seed)
    span = info.action_high - info.action_low
    actions = info.action_low + span * rng.random((count, 6), dtype=np.float32)
    if terminated is None:
        terminated = np.zeros((count, 1), dtype=np.bool_)
    if truncated is None:
        truncated = np.zeros((count, 1), dtype=np.bool_)
    return TransitionBatch(
        observations=rng.normal(size=(count, 17)).astype(np.float32),
        actions=actions.astype(np.float32),
        rewards=rng.normal(size=(count, 1)).astype(np.float32),
        next_observations=rng.normal(size=(count, 17)).astype(np.float32),
        terminated=np.asarray(terminated, dtype=np.bool_),
        truncated=np.asarray(truncated, dtype=np.bool_),
    )


def make_dataset(count: int = 64, seed: int = 0) -> ArrayOfflineDataset:
    info = make_info()
    metadata = {
        "schema_version": 1,
        "dataset_id": info.dataset_id,
        "total_episodes": 2,
        "total_transitions": count,
        "env_spec": {"id": "HalfCheetah-v5", "max_episode_steps": 1000},
        "preprocessing": {
            "observation_mean": info.observation_mean.tolist(),
            "observation_std": info.observation_std.tolist(),
            "reward_transform": "reward * 1.0 + 0.0 (unchanged)",
        },
    }
    return ArrayOfflineDataset(make_batch(count, seed=seed, info=info), info, metadata)


def test_expectile_loss_matches_the_upper_expectile_definition():
    residual = torch.tensor([1.0, -2.0, 0.0])
    # weights 0.7, 0.3, 0.7; squares 1, 4, 0; mean (0.7 + 1.2 + 0) / 3
    assert expectile_loss(residual, 0.7).item() == pytest.approx((0.7 + 1.2) / 3)


def test_bellman_target_masks_termination_and_ignores_truncation():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    next_values = torch.tensor([[6.0], [6.0], [6.0]])
    terminated = torch.tensor([[False], [True], [False]])
    target = bellman_target(rewards, next_values, terminated, discount=0.99)
    assert target[0].item() == pytest.approx(6.94)
    assert target[1].item() == pytest.approx(1.0)
    assert target[2].item() == pytest.approx(6.94)


def test_advantage_weights_match_capped_exponential():
    advantage = torch.tensor([[3.0], [-2.0]])
    weights = advantage_weights(advantage, inverse_temperature=3.0, max_weight=100.0)
    assert weights[0].item() == pytest.approx(100.0)
    assert weights[1].item() == pytest.approx(math.exp(-6.0))


def test_twin_q_and_weighted_cloning_losses_are_analytical():
    q1 = torch.tensor([[0.0], [2.0]])
    q2 = torch.tensor([[0.0], [2.0]])
    target = torch.tensor([[2.0], [2.0]])
    assert twin_q_loss(q1, q2, target).item() == pytest.approx(4.0)
    log_prob = torch.tensor([0.0, -1.0], requires_grad=True)
    weights = torch.tensor([100.0, 0.5], requires_grad=True)
    loss = weighted_behavior_cloning_loss(log_prob, weights)
    assert loss.item() == pytest.approx(0.25)
    loss.backward()
    assert weights.grad is None
    assert log_prob.grad is not None


def test_soft_update_is_a_polyak_average():
    target = nn.Linear(2, 2)
    source = nn.Linear(2, 2)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.fill_(1)
        for parameter in source.parameters():
            parameter.fill_(3)
    soft_update(target, source, tau=0.25)
    for parameter in target.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 1.5))


def test_networks_expose_independent_frozen_targets():
    torch.manual_seed(0)
    info = make_info()
    networks = build_networks(ProjectConfig(), info)
    assert set(networks) == {"q1", "q2", "value", "policy", "target_q1", "target_q2"}
    observations = torch.zeros(5, 17)
    actions = torch.zeros(5, 6)
    assert networks["q1"](observations, actions).shape == (5, 1)
    assert networks["value"](observations).shape == (5, 1)
    assert networks["policy"](observations, deterministic=True).shape == (5, 6)
    first = networks["q1"].net[0]
    assert isinstance(first, nn.Linear)
    assert first.out_features == HIDDEN_DIM
    original = networks["target_q1"].net[0].weight.detach().clone()
    with torch.no_grad():
        networks["q1"].net[0].weight.add_(1)
    assert torch.allclose(networks["target_q1"].net[0].weight, original)
    assert all(not parameter.requires_grad for parameter in networks["target_q1"].parameters())
    assert all(not parameter.requires_grad for parameter in networks["target_q2"].parameters())


def test_policy_likelihood_uses_tanh_and_affine_jacobians():
    torch.manual_seed(0)
    low = np.full(6, -2, dtype=np.float32)
    high = np.full(6, 4, dtype=np.float32)
    policy = build_networks(ProjectConfig(), make_info(action_low=low, action_high=high))["policy"]
    observations = torch.zeros(1, 17)
    midpoint = torch.as_tensor((low + high) / 2).view(1, 6)
    mean, log_std = policy.distribution_params(observations)
    unit = torch.zeros_like(mean)
    log_normal = Normal(mean, log_std.exp()).log_prob(unit)
    correction = math.log(1.0 + 1e-6) + math.log(3.0)
    expected = (log_normal - correction).sum(dim=-1)
    assert torch.allclose(policy.log_prob(observations, midpoint), expected)
    with torch.no_grad():
        policy.mean_head.weight.zero_()
        policy.mean_head.bias.zero_()
    deterministic = policy(observations, deterministic=True)
    assert torch.allclose(deterministic, midpoint.expand_as(deterministic))


def test_actions_stay_inside_bounds_and_deterministic_act_does_not_consume_rng():
    info = make_info(
        action_low=np.array([-0.2, -1, 0, -0.5, -1, 0.1], dtype=np.float32),
        action_high=np.array([0.3, 0.4, 1, 0.25, 0.5, 0.8], dtype=np.float32),
    )
    torch.manual_seed(0)
    agent = IQLAgent(ProjectConfig(), info)
    observation = np.zeros(17, dtype=np.float32)
    observation.setflags(write=False)
    before = torch.get_rng_state().clone()
    deterministic = agent.act(observation, deterministic=True)
    assert torch.equal(torch.get_rng_state(), before)
    assert deterministic.shape == (6,)
    assert deterministic.dtype == np.float32
    assert np.all(deterministic >= info.action_low)
    assert np.all(deterministic <= info.action_high)
    for _ in range(30):
        stochastic = agent.act(observation, deterministic=False)
        assert np.all(stochastic >= info.action_low)
        assert np.all(stochastic <= info.action_high)


def test_update_applies_termination_mask_and_keeps_gradients_isolated(monkeypatch):
    agent = make_agent(target_tau=1.0)
    batch = make_batch(
        2,
        terminated=np.array([[True], [False]]),
        truncated=np.array([[False], [True]]),
    )
    seen: dict[str, torch.Tensor] = {}
    real_target = bellman_target

    def spy_target(rewards, next_values, terminated, discount):
        result = real_target(rewards, next_values, terminated, discount)
        seen["terminated"] = terminated.detach().clone()
        seen["target"] = result.detach().clone()
        seen["rewards"] = rewards.detach().clone()
        seen["next_values"] = next_values.detach().clone()
        return result

    calls: list[tuple[nn.Module, ...]] = []
    real_guard = IQLAgent._assert_grads_isolated

    def spy_guard(self, *allowed):
        calls.append(allowed)
        return real_guard(self, *allowed)

    monkeypatch.setattr("iql_project.iql.bellman_target", spy_target)
    monkeypatch.setattr(IQLAgent, "_assert_grads_isolated", spy_guard)
    metrics = agent.update(batch)
    assert set(metrics) == {"q_loss", "v_loss", "policy_loss", "advantage_mean"}
    assert all(math.isfinite(value) and type(value) is float for value in metrics.values())
    alive = ~seen["terminated"]
    assert torch.equal(seen["target"][seen["terminated"]], seen["rewards"][seen["terminated"]])
    expected_alive = seen["rewards"][alive] + agent.config.discount * seen["next_values"][alive]
    assert torch.allclose(seen["target"][alive], expected_alive)
    assert len(calls) == 3
    assert agent.step == 1
    for online_name, target_name in (("q1", "target_q1"), ("q2", "target_q2")):
        for online, target in zip(
            getattr(agent, online_name).parameters(),
            getattr(agent, target_name).parameters(),
            strict=True,
        ):
            assert torch.allclose(online, target)


def test_repeated_updates_stay_finite_and_move_the_value_network():
    agent = make_agent()
    before = agent.value.net[-1].weight.detach().clone()
    batch = make_batch(16, seed=1)
    for _ in range(20):
        metrics = agent.update(batch)
        assert all(math.isfinite(value) for value in metrics.values())
    assert agent.step == 20
    assert not torch.allclose(agent.value.net[-1].weight, before)


def test_save_load_restores_actions_rng_and_dataset_info(tmp_path):
    torch.manual_seed(4)
    agent = IQLAgent(make_config(), make_info())
    sampler = np.random.default_rng(4)
    agent.bind_sampler(sampler)
    observation = np.linspace(-1, 1, 17, dtype=np.float32)
    agent.update(make_batch(4, seed=2))
    torch.manual_seed(123)
    agent.save(tmp_path / "agent.pt")
    deterministic = agent.act(observation, deterministic=True)
    stochastic = agent.act(observation, deterministic=False)

    restored_info = dataset_info_from_checkpoint(tmp_path / "agent.pt")
    restored_config = config_from_checkpoint(tmp_path / "agent.pt")
    np.testing.assert_array_equal(restored_info.observation_mean, agent.info.observation_mean)
    np.testing.assert_array_equal(restored_info.observation_std, agent.info.observation_std)
    np.testing.assert_array_equal(restored_info.action_low, agent.info.action_low)
    assert restored_config == agent.config

    torch.manual_seed(999)
    other = IQLAgent(restored_config, restored_info)
    other_sampler = np.random.default_rng(0)
    other.bind_sampler(other_sampler)
    other.load(tmp_path / "agent.pt")
    assert other.step == 1
    np.testing.assert_allclose(other.act(observation, deterministic=True), deterministic)
    np.testing.assert_allclose(other.act(observation, deterministic=False), stochastic)
    assert other_sampler.random() == pytest.approx(sampler.random())
    payload = torch.load(tmp_path / "agent.pt", map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 1
    assert payload["dataset_id"] == agent.info.dataset_id
    assert payload["optimizers"]["q"]["state"]
    assert {"python", "torch", "numpy", "iql_project"} <= set(payload["versions"])
    assert payload["git_commit"]


def test_checkpoint_rejects_a_different_dataset_contract(tmp_path):
    agent = make_agent()
    agent.save(tmp_path / "agent.pt")
    shifted = np.array(agent.info.observation_mean, copy=True)
    shifted[0] = 1
    mismatched = replace(agent.info, observation_mean=shifted)
    other = IQLAgent(ProjectConfig(), mismatched)
    with pytest.raises(ValueError, match="preprocessing"):
        other.load(tmp_path / "agent.pt")


def test_training_logs_jsonl_and_resume_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    dataset = make_dataset()
    monkeypatch.setattr(
        "iql_project.train.load_offline_dataset",
        lambda config, download=False: dataset,
    )
    short = make_config(total_steps=2, checkpoint_interval=2, seed=7)
    full = replace(short, total_steps=3)
    partial = train(short, run_dir=tmp_path / "results" / "partial")
    resumed = train(full, run_dir=tmp_path / "results" / "resumed", resume=partial)
    complete = train(full, run_dir=tmp_path / "results" / "complete")

    partial_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "partial" / "training.jsonl").read_text().splitlines()
    ]
    resumed_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "resumed" / "training.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in partial_rows] == [1, 2]
    assert [row["step"] for row in resumed_rows] == [3]
    for row in partial_rows + resumed_rows:
        assert set(row) == {
            "step",
            "q_loss",
            "v_loss",
            "policy_loss",
            "advantage_mean",
            "elapsed_seconds",
        }
        assert all(math.isfinite(row[name]) for name in row if name != "step")

    manifest = json.loads((tmp_path / "results" / "resumed" / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "resumed"
    assert manifest["dataset_id"] == dataset.info.dataset_id
    assert manifest["updates_completed"] == 3
    assert manifest["training_seed"] == 7
    assert manifest["recovered_env_spec"]["id"] == "HalfCheetah-v5"
    assert manifest["dataset_cardinality"]["transitions"] == len(dataset)
    assert manifest["finite_preprocessing"] is True
    assert manifest["resumed_from"]
    assert any(item["role"] == "checkpoint" and item["sha256"] for item in manifest["artifacts"])
    assert (tmp_path / "checkpoints" / "resumed" / "step_3.pt").is_file()
    assert Path(resumed).name == "step_3.pt"

    full_agent = IQLAgent(full, dataset.info)
    full_agent.load(complete)
    resumed_agent = IQLAgent(full, dataset.info)
    resumed_agent.load(resumed)
    assert full_agent.step == resumed_agent.step == 3
    for name in ("q1", "q2", "value", "policy", "target_q1", "target_q2"):
        full_module = getattr(full_agent, name)
        resumed_module = getattr(resumed_agent, name)
        for left, right in zip(full_module.parameters(), resumed_module.parameters(), strict=True):
            assert torch.allclose(left, right)
    observation = np.zeros(17, dtype=np.float32)
    np.testing.assert_allclose(
        full_agent.act(observation), resumed_agent.act(observation), atol=0, rtol=0
    )


def test_resume_in_the_same_directory_appends_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "iql_project.train.load_offline_dataset",
        lambda config, download=False: make_dataset(),
    )
    run_dir = tmp_path / "results" / "same"
    config = make_config(total_steps=2, checkpoint_interval=2, seed=3)
    checkpoint = train(config, run_dir=run_dir)
    continued = train(replace(config, total_steps=4), run_dir=run_dir, resume=checkpoint)
    steps = [
        json.loads(line)["step"] for line in (run_dir / "training.jsonl").read_text().splitlines()
    ]
    assert steps == [1, 2, 3, 4]
    assert Path(continued).name == "step_4.pt"
    with pytest.raises(FileExistsError):
        train(config, run_dir=run_dir)


def test_resume_rejects_a_learning_rate_the_optimizers_will_not_use(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "iql_project.train.load_offline_dataset",
        lambda config, download=False: make_dataset(),
    )
    config = make_config(total_steps=1, checkpoint_interval=1, learning_rate=3e-4, seed=0)
    checkpoint = train(config, run_dir=tmp_path / "results" / "original")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_rates = [
        group["lr"]
        for name in ("q", "v", "policy")
        for group in payload["optimizers"][name]["param_groups"]
    ]
    assert saved_rates == [3e-4, 3e-4, 3e-4]
    assert payload["config"]["learning_rate"] == 3e-4

    changed = replace(config, learning_rate=1e-3, total_steps=2)
    with pytest.raises(ValueError, match=r"learning_rate \(checkpoint 0.0003, requested 0.001\)"):
        train(changed, run_dir=tmp_path / "results" / "misrecorded", resume=checkpoint)
    assert not (tmp_path / "results" / "misrecorded").exists()


def test_resume_records_permitted_overrides_and_the_optimizer_learning_rate(tmp_path, monkeypatch):
    dataset = make_dataset()
    monkeypatch.setattr(
        "iql_project.train.load_offline_dataset",
        lambda config, download=False: dataset,
    )
    config = make_config(total_steps=1, checkpoint_interval=1, learning_rate=3e-4, seed=0)
    checkpoint = train(config, run_dir=tmp_path / "results" / "original")
    continued = replace(config, total_steps=2, checkpoint_interval=2)
    final = train(continued, run_dir=tmp_path / "results" / "continued", resume=checkpoint)
    manifest = json.loads((tmp_path / "results" / "continued" / "manifest.json").read_text())
    assert manifest["config"]["learning_rate"] == 3e-4
    assert manifest["optimizer_learning_rates"] == {"q": 3e-4, "v": 3e-4, "policy": 3e-4}
    assert manifest["resume_overrides"] == {
        "checkpoint_interval": {"checkpoint": 1, "requested": 2},
        "total_steps": {"checkpoint": 1, "requested": 2},
    }
    restored = torch.load(final, map_location="cpu", weights_only=False)
    assert restored["config"]["learning_rate"] == 3e-4
    assert [group["lr"] for group in restored["optimizers"]["q"]["param_groups"]] == [3e-4]


def test_resume_rejects_a_log_from_a_different_step(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "iql_project.train.load_offline_dataset",
        lambda config, download=False: make_dataset(),
    )
    run_dir = tmp_path / "results" / "mismatch"
    config = make_config(total_steps=2, checkpoint_interval=2, seed=1)
    checkpoint = train(config, run_dir=run_dir)
    log_path = run_dir / "training.jsonl"
    log_path.write_text(log_path.read_text().splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training.jsonl"):
        train(replace(config, total_steps=3), run_dir=run_dir, resume=checkpoint)


def test_train_cli_exposes_resume_and_evaluate_stays_unimplemented(tmp_path):
    help_result = subprocess.run(
        [sys.executable, "-m", "iql_project", "train", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--resume" in help_result.stdout
    assert "--run-dir" in help_result.stdout
    assert "--download" in help_result.stdout

    missing = subprocess.run(
        [
            sys.executable,
            "-m",
            "iql_project",
            "train",
            "--resume",
            str(tmp_path / "missing.pt"),
            "--run-dir",
            str(tmp_path / "results" / "run"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "resume checkpoint not found" in missing.stderr.lower()

    evaluate = subprocess.run(
        [sys.executable, "-m", "iql_project", "evaluate"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert evaluate.returncode == 2
    assert "not implemented" in evaluate.stderr.lower()
    assert not list(tmp_path.iterdir())


@pytest.mark.real_data
def test_one_thousand_updates_on_real_minari_data(tmp_path):
    if os.environ.get("IQL_RUN_REAL_DATA") != "1":
        pytest.skip("Set IQL_RUN_REAL_DATA=1 to train 1000 updates on the Minari dataset")
    config = replace(
        load_real_config(),
        total_steps=1000,
        checkpoint_interval=1000,
        seed=0,
        device="cpu",
    )
    checkpoint = train(
        config,
        run_dir=tmp_path / "results" / "real_1000",
        download=True,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "real_1000" / "training.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1000
    assert rows[-1]["step"] == 1000
    for row in rows:
        assert all(
            math.isfinite(row[name])
            for name in ("q_loss", "v_loss", "policy_loss", "advantage_mean")
        )
    assert Path(checkpoint).is_file()


def load_real_config() -> ProjectConfig:
    from iql_project.config import load_config

    return load_config(ROOT / "configs/halfcheetah.toml")
