"""A repeated replay must compare individual episodes, not only rounded means."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


def load_replay_module():
    path = Path(__file__).resolve().parents[1] / "scripts/replay_published.py"
    spec = importlib.util.spec_from_file_location("replay_published", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_checkpoint_download_checks_bytes_before_returning_path(tmp_path):
    module = load_replay_module()
    download = getattr(module, "download_checkpoint", None)
    assert callable(download), "Verified continuation checkpoint download is missing"
    source = tmp_path / "source.pt"
    source.write_bytes(b"trusted checkpoint bytes")
    target = tmp_path / "downloaded.pt"
    digest = hashlib.sha256(b"trusted checkpoint bytes").hexdigest()
    assert download(source.as_uri(), digest, target) == target
    assert target.read_bytes() == b"trusted checkpoint bytes"


@pytest.mark.parametrize("cached", [False, True])
def test_corrupted_checkpoint_is_rejected_in_download_and_cache(tmp_path, cached):
    module = load_replay_module()
    download = getattr(module, "download_checkpoint", None)
    assert callable(download), "Verified continuation checkpoint download is missing"
    source = tmp_path / "source.pt"
    source.write_bytes(b"trusted checkpoint bytes" if cached else b"wrong bytes")
    target = tmp_path / "downloaded.pt"
    if cached:
        target.write_bytes(b"wrong bytes")
    digest = hashlib.sha256(b"trusted checkpoint bytes").hexdigest()
    with pytest.raises(ValueError, match="checksum mismatch"):
        download(source.as_uri(), digest, target)
    if not cached:
        assert not target.exists(), "Unverified download must not become the cached checkpoint"


def test_interrupted_download_can_retry_without_a_poisoned_cache(tmp_path, monkeypatch):
    module = load_replay_module()
    source = tmp_path / "source.pt"
    source.write_bytes(b"trusted checkpoint bytes")
    target = tmp_path / "downloaded.pt"
    digest = hashlib.sha256(b"trusted checkpoint bytes").hexdigest()

    def interrupted_transfer(url, filename):
        Path(filename).write_bytes(b"partial")
        raise OSError("Connection interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(module.urllib.request, "urlretrieve", interrupted_transfer)
        with pytest.raises(OSError, match="Connection interrupted"):
            module.download_checkpoint(source.as_uri(), digest, target)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["source.pt"]
    assert (
        module.download_checkpoint(source.as_uri(), digest, target).read_bytes()
        == b"trusted checkpoint bytes"
    )


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
