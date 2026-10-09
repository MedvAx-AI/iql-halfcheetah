"""A repeated replay must compare individual episodes, not only rounded means."""

import importlib.util
from pathlib import Path

import pytest


def test_repeat_comparison_ignores_run_name_but_detects_changed_episode():
    path = Path(__file__).resolve().parents[1] / "scripts/replay_published.py"
    spec = importlib.util.spec_from_file_location("replay_published", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = [{"run_id": "first", "episode_return": 12.5, "episode_length": 1000}]
    repeated = [{"run_id": "repeat", "episode_return": 12.5, "episode_length": 1000}]
    module.verify_repeat(first, repeated)
    with pytest.raises(ValueError, match="Repeated evaluation"):
        module.verify_repeat(first, [{**repeated[0], "episode_return": 12.6}])
    with pytest.raises(ValueError, match="Repeated evaluation"):
        module.verify_repeat(first, [])
