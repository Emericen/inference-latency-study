from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.runner import run_config_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one inference latency study config.")
    parser.add_argument("--config", required=True, help="Path to config YAML file.")
    parser.add_argument("--output", help="Path to output JSONL file.")
    parser.add_argument("--base-url", help="Override base URL.")
    parser.add_argument(
        "--client-mode",
        choices=["local", "remote"],
        help="Override client mode metadata.",
    )
    parser.add_argument("--region", help="Override region metadata.")
    parser.add_argument("--gpu-type", help="Override GPU type metadata.")
    parser.add_argument("--gpu-count", type=int, help="Override GPU count metadata.")
    parser.add_argument("--tp-size", type=int, help="Override tensor parallel size metadata.")
    parser.add_argument("--runs", type=int, help="Override measured runs per case.")
    parser.add_argument("--warmups", type=int, help="Override warmups per case.")
    parser.add_argument(
        "--prepared-index-offset",
        type=int,
        help="Offset into the prepared image manifest for prepared-image payloads.",
    )
    parser.add_argument(
        "--wait-for-server-timeout-s",
        type=float,
        default=30.0,
        help="How long to wait for /v1/models before failing.",
    )
    args = parser.parse_args()

    overrides = {}
    if args.base_url:
        overrides["base_url"] = args.base_url
    if args.client_mode:
        overrides["client_mode"] = args.client_mode
    if args.region:
        overrides["region"] = args.region
    if args.gpu_type:
        overrides["gpu_type"] = args.gpu_type
    if args.gpu_count is not None:
        overrides["gpu_count"] = args.gpu_count
    if args.tp_size is not None:
        overrides["tp_size"] = args.tp_size
    if args.runs is not None:
        overrides["runs"] = args.runs
    if args.warmups is not None:
        overrides["warmups"] = args.warmups
    if args.prepared_index_offset is not None:
        overrides["payload"] = {"prepared_index_offset": args.prepared_index_offset}

    out = run_config_file(
        path=args.config,
        output_path=args.output,
        overrides=overrides or None,
        wait_for_server_timeout_s=args.wait_for_server_timeout_s,
    )
    print(out)


if __name__ == "__main__":
    main()
