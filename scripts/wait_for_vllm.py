from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.vllm import wait_until_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for a vLLM OpenAI-compatible server.")
    parser.add_argument("--base-url", required=True, help="Base URL such as http://localhost:8000/v1")
    parser.add_argument("--timeout-s", type=float, default=900.0, help="Maximum wait time.")
    args = parser.parse_args()
    wait_until_ready(args.base_url, timeout_s=args.timeout_s)
    print(args.base_url)


if __name__ == "__main__":
    main()
