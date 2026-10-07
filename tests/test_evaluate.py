"""Person 4 rollout correctness, seeded baselines, MP4 and artifact integration."""

import csv
import json
import random
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.envs.registration import EnvSpec
from moviepy import VideoFileClip

from iql_project import evaluate as evaluation
from iql_project import reporting
from iql_project.config import ProjectConfig
from iql_project.contracts import DatasetInfo
from iql_project.iql import IQLAgent


@pytest.fixture
def info():
    return DatasetInfo(
        "mujoco/halfcheetah/medium-v0",
        17,
        6,
        np.full(6, -1, dtype=np.float32),
        np.ones(6, dtype=np.float32),
        np.full(17, 10, dtype=np.float32),
        np.full(17, 2, dtype=np.float32),
        7.0,
        -9.0,  # Evaluation must still report raw rewards.
    )


class TinyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, *, end="truncated", render_mode=None, fail=False):
        self.render_mode, self.end, self.fail = render_mode, end, fail
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
        self.spec = EnvSpec("HalfCheetah-v5", max_episode_steps=3)
        self.closed = False
        self.seeds, self.actions = [], []
        self.index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seeds.append(seed)
        self.index = 0
        return np.full(17, 14, dtype=np.float64), {}

    def step(self, action):
        if self.fail:
            raise RuntimeError("step failed")
        self.actions.append(action.copy())
        reward = (2.5, -1.0, 4.0)[self.index]
        self.index += 1
        ended = self.index == 3
        return (
            np.full(17, 14 + 2 * self.index, dtype=np.float64),
            reward,
            ended and self.end in {"terminated", "both"},
            ended and self.end in {"truncated", "both"},
            {},
        )

    def render(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = self.index * 50
        return frame

    def close(self):
        self.closed = True


class SpyPolicy:
    def __init__(self):
        self.observations, self.deterministic = [], []

    def act(self, observation, *, deterministic=True):
        self.observations.append(observation.copy())
        self.deterministic.append(deterministic)
        return np.zeros(6, dtype=np.float32)


@pytest.mark.parametrize("end", ["terminated", "truncated", "both"])
def test_rollout_raw_sum_flags_saved_preprocessing_and_closure(monkeypatch, info, end):
    env = TinyEnv(end=end)
    monkeypatch.setattr(evaluation, "make_evaluation_env", lambda config: env)
    policy = SpyPolicy()
    result = evaluation.evaluate(policy, ProjectConfig(eval_episodes=2, eval_seed=123), info)
    assert result.episode_returns == (5.5, 5.5)
    assert result.episode_lengths == (3, 3)
    assert result.episode_seeds == (123, 124)
    assert env.seeds == [123, 124] and env.closed
    assert all(policy.deterministic)
    for observation, value in zip(policy.observations, (2, 3, 4, 2, 3, 4), strict=True):
        np.testing.assert_array_equal(observation, np.full(17, value, dtype=np.float32))
        assert observation.dtype == np.float32


def test_random_baseline_resets_its_stream_for_each_seed(monkeypatch, info):
    environments = []

    def factory(config):
        env = TinyEnv()
        environments.append(env)
        return env

    monkeypatch.setattr(evaluation, "make_evaluation_env", factory)
    policy = evaluation.RandomPolicy(info)
    config = ProjectConfig(eval_episodes=2)
    first = evaluation.evaluate(policy, config, info)
    policy.rng.random(99)  # Prior consumption must have no effect on the next evaluation.
    second = evaluation.evaluate(policy, config, info)
    assert first == second
    np.testing.assert_array_equal(environments[0].actions, environments[1].actions)
    assert not np.array_equal(environments[0].actions[:3], environments[0].actions[3:])


def test_seeded_environment_and_policy_repeatability(monkeypatch, info):
    class SeededEnv(TinyEnv):
        def reset(self, *, seed=None, options=None):
            observation, metadata = super().reset(seed=seed, options=options)
            return observation + self.np_random.normal(size=17), metadata

        def step(self, action):
            observation, reward, terminated, truncated, metadata = super().step(action)
            return observation, reward + float(action.sum()), terminated, truncated, metadata

    monkeypatch.setattr(evaluation, "make_evaluation_env", lambda config: SeededEnv())
    policy = IQLAgent(ProjectConfig(), info)
    first = evaluation.evaluate(policy, ProjectConfig(eval_episodes=3), info)
    assert first == evaluation.evaluate(policy, ProjectConfig(eval_episodes=3), info)
    different = evaluation.evaluate(policy, ProjectConfig(eval_episodes=3, eval_seed=20), info)
    assert first.episode_returns != different.episode_returns


def test_environment_closes_on_failed_policy_and_step(monkeypatch, info):
    env = TinyEnv(fail=True)
    monkeypatch.setattr(evaluation, "make_evaluation_env", lambda config: env)
    with pytest.raises(RuntimeError, match="step failed"):
        evaluation.evaluate(SpyPolicy(), ProjectConfig(), info)
    assert env.closed


@pytest.mark.parametrize("action", [np.zeros(5), np.full(6, np.nan), np.full(6, 2.0)])
def test_invalid_actions_rejected_and_environment_closed(monkeypatch, info, action):
    env = TinyEnv()
    monkeypatch.setattr(evaluation, "make_evaluation_env", lambda config: env)
    policy = SpyPolicy()
    policy.act = lambda observation, deterministic: action
    with pytest.raises(ValueError, match="Policy action"):
        evaluation.evaluate(policy, ProjectConfig(), info)
    assert env.closed


def test_invalid_normalization_fails_before_environment_creation(monkeypatch, info):
    def forbidden(config):
        pytest.fail("Environment created for invalid checkpoint preprocessing")

    monkeypatch.setattr(evaluation, "make_evaluation_env", forbidden)
    with pytest.raises(ValueError, match="std"):
        evaluation.evaluate(
            SpyPolicy(), ProjectConfig(), replace(info, observation_std=np.zeros(17))
        )


def test_record_video_writes_decodable_mp4_and_protects_existing_file(monkeypatch, info, tmp_path):
    env = TinyEnv(render_mode="rgb_array")
    monkeypatch.setattr(evaluation, "make_evaluation_env", lambda config, render_mode: env)
    output = evaluation.record_video(
        SpyPolicy(), ProjectConfig(), info, output_dir=tmp_path, seed=42
    )
    assert output == tmp_path / "demo_seed_42.mp4"
    assert env.closed and env.seeds == [42]
    with VideoFileClip(str(output)) as clip:
        assert clip.duration > 0
        assert clip.get_frame(0).shape == (32, 32, 3)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.record_video(SpyPolicy(), ProjectConfig(), info, output_dir=tmp_path, seed=42)
    assert output.read_bytes() == before


def test_failed_video_closes_and_leaves_no_final_mp4(monkeypatch, info, tmp_path):
    env = TinyEnv(render_mode="rgb_array", fail=True)
    monkeypatch.setattr(evaluation, "make_evaluation_env", lambda config, render_mode: env)
    with pytest.raises(RuntimeError, match="step failed"):
        evaluation.record_video(SpyPolicy(), ProjectConfig(), info, output_dir=tmp_path, seed=42)
    assert env.closed
    assert not list(tmp_path.iterdir())


def test_checkpoint_evaluation_upserts_preserves_training_manifest_and_rng(
    monkeypatch, info, tmp_path
):
    monkeypatch.setattr(evaluation, "make_evaluation_env", lambda config: TinyEnv())
    monkeypatch.setattr(reporting, "make_evaluation_env", lambda config: TinyEnv())
    checkpoint = tmp_path / "step_0.pt"
    IQLAgent(ProjectConfig(eval_episodes=2), info).save(checkpoint)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"training_seed": 0, "training_evidence": 123})
    )
    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)
    expected_torch = torch.get_rng_state().clone()
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    summary_path = reporting.evaluate_checkpoint(checkpoint, run_dir=run_dir)
    assert torch.equal(expected_torch, torch.get_rng_state())
    assert expected_python == random.getstate()
    np.testing.assert_array_equal(expected_numpy[1], np.random.get_state()[1])
    reporting.evaluate_checkpoint(checkpoint, run_dir=run_dir)
    rows = reporting.read_evaluation(run_dir / "evaluation.csv")
    assert len(rows) == 4  # Two agent episodes + two baseline episodes, without duplicates.
    assert {row["policy_name"] for row in rows} == {"iql", "random"}
    summary = json.loads(summary_path.read_text())
    assert summary["within_run"][0]["return_mean"] == 5.5
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["training_evidence"] == 123
    assert manifest["evaluation_seeds"] == [10000, 10001]
    assert manifest["evaluation_checkpoints"]["0"]["sha256"]
    assert (run_dir / "plots/evaluation_returns.png").is_file()
    with pytest.raises(ValueError, match="different preprocessing/config/evaluation seeds"):
        reporting.evaluate_checkpoint(checkpoint, run_dir=run_dir, eval_seed=42)


def make_run(root, seed, returns, *, eval_seeds=None, step=100):
    path = root / f"seed_{seed}"
    path.mkdir()
    eval_seeds = list(range(10000, 10000 + len(returns))) if eval_seeds is None else eval_seeds
    manifest = {
        "run_id": path.name,
        "recovered_env_spec": {"id": "HalfCheetah-v5"},
        "evaluation_protocol": {
            "training_seed": seed,
            "training_config": {"seed": seed},
            "evaluation_seeds": eval_seeds,
            "transforms": {},
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    with (path / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=reporting.CSV_FIELDS)
        writer.writeheader()
        for policy in ("iql", "random"):
            for index, (reward, eval_seed) in enumerate(zip(returns, eval_seeds, strict=True)):
                writer.writerow(
                    dict(
                        zip(
                            reporting.CSV_FIELDS,
                            (
                                path.name,
                                step if policy == "iql" else 0,
                                seed,
                                eval_seed,
                                index,
                                reward if policy == "iql" else -1,
                                1000,
                                policy,
                            ),
                            strict=True,
                        )
                    )
                )
    return path


def test_summary_uses_run_means_and_separates_episode_variation(tmp_path):
    runs = [make_run(tmp_path, seed, [mean - 1, mean + 1]) for seed, mean in enumerate((2, 11, 21))]
    output = reporting.summarize_runs(runs, output_dir=tmp_path / "aggregate")
    result = json.loads(output.read_text())
    across = result["across_runs"][0]
    assert across["mean_of_run_means"] == pytest.approx(34 / 3)
    assert across["std_sample_of_run_means"] == pytest.approx(np.std([2, 11, 21], ddof=1))
    assert all(
        row["return_std_population"] == 1
        for row in result["within_run"]
        if row["policy_name"] == "iql"
    )
    assert not result["minimum_seed_episode_protocol_met"]


def test_three_seeds_ten_episodes_meet_protocol(tmp_path):
    runs = [make_run(tmp_path, seed, [seed] * 10) for seed in range(3)]
    output = reporting.summarize_runs(runs, output_dir=tmp_path / "aggregate")
    assert json.loads(output.read_text())["minimum_seed_episode_protocol_met"]


def test_summary_rejects_mismatched_seeds_and_duplicate_training_runs(tmp_path):
    one = make_run(tmp_path, 0, [1, 2])
    two = make_run(tmp_path, 1, [3, 4], eval_seeds=[42, 43])
    with pytest.raises(ValueError, match="share environment"):
        reporting.summarize_runs([one, two], output_dir=tmp_path / "aggregate")
    with pytest.raises(ValueError, match="distinct training seeds"):
        reporting.summarize_runs([one, one], output_dir=tmp_path / "aggregate")


def test_mismatched_final_steps_are_not_claimed_complete(tmp_path):
    runs = [make_run(tmp_path, seed, [seed] * 10, step=100 + seed) for seed in range(3)]
    output = reporting.summarize_runs(runs, output_dir=tmp_path / "aggregate")
    result = json.loads(output.read_text())
    assert not result["minimum_seed_episode_protocol_met"]
    assert all(
        row["runs"] == 1 and row["std_sample_of_run_means"] is None for row in result["across_runs"]
    )


def test_summary_rejects_missing_episode_instead_of_claiming_protocol(tmp_path):
    runs = [make_run(tmp_path, seed, [seed] * 10) for seed in range(3)]
    csv_path = runs[0] / "evaluation.csv"
    lines = csv_path.read_text().splitlines(keepends=True)
    csv_path.write_text("".join(lines[:-1]))
    with pytest.raises(ValueError, match="exactly the fixed evaluation seed set"):
        reporting.summarize_runs(runs, output_dir=tmp_path / "aggregate")


def test_checkpoint_failure_is_recorded_and_environment_closed(monkeypatch, info, tmp_path):
    checkpoint = tmp_path / "step_0.pt"
    IQLAgent(ProjectConfig(), info).save(checkpoint)
    monkeypatch.setattr(reporting, "make_evaluation_env", lambda config: TinyEnv())
    env = TinyEnv(fail=True)
    monkeypatch.setattr(evaluation, "make_evaluation_env", lambda config: env)
    run_dir = tmp_path / "failed_run"
    with pytest.raises(RuntimeError, match="step failed"):
        reporting.evaluate_checkpoint(checkpoint, run_dir=run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "step failed" in manifest["failed_evaluations"][0]["error"]
    assert env.closed and not (run_dir / "summary.json").exists()
