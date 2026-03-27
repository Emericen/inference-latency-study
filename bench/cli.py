from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference Latency Study CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run one benchmark scenario file")
    run_parser.add_argument(
        "--scenario", required=True, help="Path to scenario YAML file"
    )
    run_parser.add_argument("--output", help="Path to output JSONL file")
    run_parser.add_argument("--base-url", help="Override backend base URL")
    run_parser.add_argument(
        "--client-mode",
        choices=["local", "remote"],
        help="Override client mode metadata",
    )
    run_parser.add_argument("--region", help="Override region metadata")
    run_parser.add_argument("--gpu-type", help="Override GPU type metadata")
    run_parser.add_argument("--gpu-count", type=int, help="Override GPU count metadata")
    run_parser.add_argument(
        "--tp-size", type=int, help="Override tensor parallel size metadata"
    )
    run_parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Launch and manage a local vLLM server for the scenario run",
    )
    run_parser.add_argument(
        "--server-host",
        default="127.0.0.1",
        help="Host to bind the managed local vLLM server",
    )
    run_parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="Port for the managed local vLLM server",
    )
    run_parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=900.0,
        help="Maximum time to wait for vLLM readiness",
    )
    run_parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Do not stop the managed local vLLM server after the run finishes",
    )
    run_parser.add_argument(
        "--server-log-dir",
        help="Directory for managed vLLM server logs",
    )
    run_parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Additional argument to pass through to `vllm serve`",
    )

    agg_parser = sub.add_parser("aggregate", help="Aggregate a raw JSONL result file")
    agg_parser.add_argument("--input", required=True, help="Path to raw JSONL results")
    agg_parser.add_argument(
        "--output", required=True, help="Path to summary CSV output"
    )

    args = parser.parse_args()

    if args.command == "run":
        from bench.preflight import ensure_run_dependencies
        from bench.runner import run_scenario_file

        ensure_run_dependencies(launch_server=args.launch_server)

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

        out = run_scenario_file(
            path=args.scenario,
            output_path=args.output,
            overrides=overrides or None,
            launch_server=args.launch_server,
            server_host=args.server_host,
            server_port=args.server_port,
            startup_timeout_s=args.startup_timeout_s,
            keep_server=args.keep_server,
            server_log_dir=args.server_log_dir,
            server_args=args.server_arg or None,
        )
        print(out)
        return

    if args.command == "aggregate":
        from analysis.aggregate import aggregate_jsonl

        out = aggregate_jsonl(input_path=args.input, output_path=args.output)
        print(out)
        return


if __name__ == "__main__":
    main()
