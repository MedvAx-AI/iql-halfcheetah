"""Keep notebook state in one locked project kernel, separate from Colab's imports."""

import atexit
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from jupyter_client import KernelManager


class ProjectKernel:
    def __init__(self, root: Path, python: str) -> None:
        self.config = TemporaryDirectory(prefix="iql-notebook-kernel-")
        env = dict(
            os.environ,
            IPYTHONDIR=self.config.name,
            JUPYTER_CONFIG_DIR=self.config.name,
            PYTHONPATH=str(root / "src"),
        )
        self.manager = KernelManager(kernel_name="python3")
        self.manager.kernel_spec.argv = [
            str(python),
            "-m",
            "ipykernel_launcher",
            "--IPKernelApp.kernel_class=ipykernel.ipkernel.IPythonKernel",
            "--InteractiveShellApp.extensions=[]",
            "--InteractiveShellApp.extra_extensions=[]",
            "-f",
            "{connection_file}",
        ]
        try:
            self.manager.start_kernel(cwd=str(root), env=env)
            self.client = self.manager.client()
            self.client.start_channels()
            self.client.wait_for_ready(timeout=120)
            self.execute("from pathlib import Path\nROOT = Path.cwd()")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _output(message: dict) -> None:
        from IPython.display import clear_output, display, update_display

        kind, content = message["msg_type"], message["content"]
        if kind == "stream":
            print(
                content["text"],
                end="",
                file=sys.stderr if content["name"] == "stderr" else sys.stdout,
            )
        elif kind in {"display_data", "execute_result", "update_display_data"}:
            display_id = content.get("transient", {}).get("display_id")
            if kind == "update_display_data" and display_id:
                update_display(content["data"], raw=True, display_id=display_id)
            else:
                display(
                    content["data"],
                    raw=True,
                    metadata=content.get("metadata", {}),
                    display_id=display_id,
                )
        elif kind == "clear_output":
            clear_output(wait=content.get("wait", False))

    def execute(self, code: str) -> None:
        reply = self.client.execute_interactive(
            code, timeout=1800, store_history=False, allow_stdin=False, output_hook=self._output
        )
        content = reply["content"]
        if content["status"] != "ok":
            raise RuntimeError(
                f"Project cell failed: {content.get('ename')}: {content.get('evalue')}"
            )

    def close(self) -> None:
        if hasattr(self, "client"):
            self.client.stop_channels()
        try:
            if self.manager.has_kernel:
                self.manager.shutdown_kernel(now=True)
        finally:
            self.config.cleanup()


def register_project_magic(root: Path, python: str) -> ProjectKernel:
    from IPython import get_ipython

    shell = get_ipython()
    previous = shell.user_ns.get("_project_runtime")
    if previous is not None:
        previous.close()
    runtime = ProjectKernel(root, python)
    shell.user_ns["_project_runtime"] = runtime

    def project(line: str, cell: str) -> None:
        runtime.execute(cell)

    shell.register_magic_function(project, magic_kind="cell", magic_name="project")
    atexit.register(runtime.close)
    return runtime
