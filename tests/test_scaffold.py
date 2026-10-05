"""Integration guarantees; algorithm correctness belongs to the role PRs."""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_matches_environment_contract():
    from iql_project.config import load_config

    config = load_config(ROOT / "configs/halfcheetah.toml")
    assert config.env_id == "HalfCheetah-v5"
    assert config.dataset_id == "mujoco/halfcheetah/medium-v0"
    assert (config.observation_dim, config.action_dim) == (17, 6)
    assert config.eval_seed != config.seed


@pytest.mark.parametrize(
    "override",
    [
        {"discount": True},
        {"discount": 1.1},
        {"expectile": 0.5},
        {"learning_rate": float("nan")},
        {"batch_size": 0},
        {"total_steps": 1.5},
        {"device": "invalid"},
    ],
)
def test_invalid_config_fails_before_any_run(override):
    from iql_project.config import ProjectConfig

    with pytest.raises(ValueError):
        ProjectConfig(**override)


def test_unknown_config_key_is_rejected(tmp_path):
    from iql_project.config import load_config

    config_file = tmp_path / "typo.toml"
    config_file.write_text("batch_szie = 256\n", encoding="utf-8")
    with pytest.raises(ValueError, match="batch_szie"):
        load_config(config_file)


def test_batch_contract_preserves_two_ending_flags():
    from iql_project.contracts import TransitionBatch

    batch = TransitionBatch(
        observations=np.zeros((2, 17), dtype=np.float32),
        actions=np.zeros((2, 6), dtype=np.float32),
        rewards=np.zeros((2, 1), dtype=np.float32),
        next_observations=np.ones((2, 17), dtype=np.float32),
        terminated=np.array([[False], [True]]),
        truncated=np.array([[True], [False]]),
    )
    assert batch.terminated[0, 0] != batch.truncated[0, 0]
    assert batch.next_observations.shape == batch.observations.shape


def test_public_interfaces_import_without_side_effects():
    for name in ("dataset", "environment", "networks", "iql", "train", "evaluate"):
        assert importlib.import_module(f"iql_project.{name}")


def test_scaffold_cli_reports_ownership_and_contract():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "iql_project",
            "check",
            "--config",
            str(ROOT / "configs/halfcheetah.toml"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "configuration valid" in result.stdout.lower()
    assert "not implemented" in result.stdout.lower()
    assert "HalfCheetah-v5" in result.stdout
    assert "Person 3" in result.stdout


def test_evaluate_command_fails_without_artifacts(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "iql_project", "evaluate"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "not implemented" in result.stderr.lower()
    assert not list(tmp_path.iterdir())
