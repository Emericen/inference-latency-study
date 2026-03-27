from __future__ import annotations

import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def is_ready(base_url: str) -> bool:
    url = f"{base_url.rstrip('/')}/models"
    try:
        with urlopen(url, timeout=5) as response:
            return response.status == 200
    except (HTTPError, URLError, TimeoutError):
        return False


def wait_until_ready(base_url: str, timeout_s: float = 900.0, poll_interval_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if is_ready(base_url):
            return
        time.sleep(poll_interval_s)
    raise RuntimeError(
        f"Timed out waiting for vLLM readiness at `{base_url.rstrip('/')}/models`."
    )
