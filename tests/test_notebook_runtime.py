"""The notebook kernel must preserve state, forward output and propagate failures."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def runtime(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/notebook_runtime.py"
    spec = importlib.util.spec_from_file_location("notebook_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.ProjectKernel(tmp_path, sys.executable)
    yield kernel
    kernel.close()


def test_project_cells_share_state_and_working_directory(runtime, tmp_path, capsys):
    runtime.execute("from pathlib import Path\nx = 41\nprint(Path.cwd())")
    runtime.execute("x += 1\nprint(x)")
    output = capsys.readouterr().out
    assert str(tmp_path) in output
    assert "42" in output


def test_project_cell_error_is_not_silently_skipped(runtime, capsys):
    with pytest.raises(RuntimeError, match="ValueError.*bad transition"):
        runtime.execute("raise ValueError('bad transition')\nprint('wrongly continued')")
    assert "wrongly continued" not in capsys.readouterr().out
    runtime.execute("print('kernel still usable')")
    assert "kernel still usable" in capsys.readouterr().out


def test_project_rich_output_reaches_notebook_display(runtime, monkeypatch):
    import IPython.display

    bundles = []
    monkeypatch.setattr(IPython.display, "display", lambda bundle, **kwargs: bundles.append(bundle))
    runtime.execute("from IPython.display import display, HTML\ndisplay(HTML('<b>result</b>'))")
    assert any(bundle.get("text/html") == "<b>result</b>" for bundle in bundles)


def test_project_display_updates_keep_the_initial_display_id(runtime, monkeypatch):
    import IPython.display

    initial, updates = [], []
    monkeypatch.setattr(IPython.display, "display", lambda data, **kw: initial.append((data, kw)))
    monkeypatch.setattr(
        IPython.display, "update_display", lambda data, **kw: updates.append((data, kw))
    )
    runtime.execute(
        "from IPython.display import display, HTML\n"
        "handle = display(HTML('<b>initial</b>'), display_id='progress')\n"
        "handle.update(HTML('<b>done</b>'))"
    )
    assert initial[0][1].get("display_id") == "progress"
    assert updates[0][1]["display_id"] == "progress"
    assert updates[0][0]["text/html"] == "<b>done</b>"
