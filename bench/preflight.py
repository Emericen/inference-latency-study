from __future__ import annotations

import importlib
import shutil


def ensure_run_dependencies(*, launch_server: bool) -> None:
    importlib.import_module("openai")

    if launch_server and shutil.which("vllm") is None:
        raise RuntimeError(
            "Could not find the `vllm` executable in PATH. "
            "Install it in the active environment before using `--launch-server`."
        )
