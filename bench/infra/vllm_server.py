from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


@dataclass
class ManagedVLLMServer:
    process: subprocess.Popen
    model: str
    base_url: str
    log_path: Path

    def stop(self) -> None:
        if self.process.poll() is not None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


def start_vllm_server(
    *,
    model: str,
    host: str,
    port: int,
    startup_timeout_s: float,
    log_path: Path,
    extra_args: Sequence[str] | None = None,
) -> ManagedVLLMServer:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")

    command = [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if extra_args:
        command.extend(extra_args)

    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base_url = f"http://{host}:{port}/v1"
    ready_url = f"{base_url}/models"
    deadline = time.monotonic() + startup_timeout_s

    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_file.close()
            raise RuntimeError(
                f"vLLM exited before becoming ready for model `{model}`. "
                f"See log: {log_path}"
            )

        if _is_ready(ready_url):
            log_file.close()
            return ManagedVLLMServer(
                process=process,
                model=model,
                base_url=base_url,
                log_path=log_path,
            )
        time.sleep(2.0)

    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    log_file.close()
    raise RuntimeError(
        f"Timed out waiting for vLLM to become ready for model `{model}`. "
        f"See log: {log_path}"
    )


def _is_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=5) as response:
            return response.status == 200
    except (HTTPError, URLError, TimeoutError):
        return False
