from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from bench.backends.openai_compat import OpenAICompatBackend
from bench.infra.vllm_server import ManagedVLLMServer, start_vllm_server
from bench.payloads.builders import build_payload
from bench.schema import ResultRow
from bench.token_budget import load_token_tools


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_backend(case: dict):
    backend_name = case["backend"]
    if backend_name != "openai_compat":
        raise NotImplementedError(f"Unsupported backend: {backend_name}")

    api_key = case.get("api_key") or os.environ.get(
        case.get("api_key_env", ""), "EMPTY"
    )
    if case.get("api_key_env") == "" and not case.get("api_key"):
        api_key = "EMPTY"
    return OpenAICompatBackend(
        base_url=case["base_url"],
        api_key=api_key,
        timeout_s=case.get("timeout_s", 300.0),
    )


def run_scenario_file(
    path: str,
    output_path: str | None = None,
    overrides: dict | None = None,
    launch_server: bool = False,
    server_host: str = "127.0.0.1",
    server_port: int = 8000,
    startup_timeout_s: float = 900.0,
    keep_server: bool = False,
    server_log_dir: str | None = None,
    server_args: list[str] | None = None,
) -> Path:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text())

    experiment_name = config["experiment_name"]
    defaults = config.get("defaults", {})
    if overrides:
        defaults = _deep_merge(defaults, overrides)
    cases = config["cases"]

    if output_path is None:
        out_dir = Path("results/raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{config_path.stem}.jsonl"
    else:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

    active_server: ManagedVLLMServer | None = None
    active_server_model: str | None = None
    log_dir = Path(server_log_dir) if server_log_dir else Path("results/logs")

    try:
        with output.open("w", encoding="utf-8") as f:
            for case in cases:
                merged = _deep_merge(defaults, case)

                if launch_server:
                    model = merged["model"]
                    if active_server is None or active_server_model != model:
                        if active_server is not None:
                            active_server.stop()
                            active_server = None
                            active_server_model = None

                        log_name = model.replace("/", "__").replace(":", "_")
                        log_path = log_dir / f"{config_path.stem}__{log_name}.log"
                        active_server = start_vllm_server(
                            model=model,
                            host=server_host,
                            port=server_port,
                            startup_timeout_s=startup_timeout_s,
                            log_path=log_path,
                            extra_args=server_args,
                        )
                        active_server_model = model

                    merged["base_url"] = active_server.base_url

                backend = _build_backend(merged)
                token_tools = load_token_tools(
                    merged["token_budget_model"],
                    local_files_only=merged.get("local_files_only", True),
                )

                total_runs = merged["warmups"] + merged["runs"]
                for request_index in range(total_runs):
                    is_warmup = request_index < merged["warmups"]
                    seed = (
                        merged["seed"]
                        if merged["payload"]["cache_mode"] == "repeat"
                        else merged["seed"] + request_index
                    )
                    payload = build_payload(
                        tokenizer=token_tools.tokenizer,
                        image_processor=token_tools.image_processor,
                        payload_kind=merged["payload"]["kind"],
                        target_prompt_tokens=merged["payload"]["target_prompt_tokens"],
                        question=merged["payload"]["question"],
                        seed=seed,
                        image_width=merged["payload"].get("image_width", 1920),
                        image_height=merged["payload"].get("image_height", 1080),
                    )
                    response = backend.run_request(
                        model=merged["model"],
                        messages=payload.messages,
                        max_completion_tokens=merged["max_completion_tokens"],
                    )

                    completion_tokens = len(
                        token_tools.tokenizer(
                            response.content,
                            add_special_tokens=False,
                        )["input_ids"]
                    )
                    decode_tps = None
                    if response.ttft_s is not None and completion_tokens > 1:
                        decode_window = response.total_latency_s - response.ttft_s
                        if decode_window > 0:
                            decode_tps = (completion_tokens - 1) / decode_window

                    row = ResultRow(
                        experiment_name=experiment_name,
                        scenario_id=merged["scenario_id"],
                        case_id=merged["case_id"],
                        request_index=request_index,
                        is_warmup=is_warmup,
                        model_id=merged["model"],
                        arch_type=merged["arch_type"],
                        precision=merged["precision"],
                        server_mode=merged["server_mode"],
                        client_mode=merged["client_mode"],
                        payload_kind=merged["payload"]["kind"],
                        cache_mode=merged["payload"]["cache_mode"],
                        prompt_tokens=payload.prompt_tokens,
                        completion_tokens=completion_tokens,
                        image_count=payload.image_count,
                        image_bytes=payload.image_bytes,
                        ttft_s=response.ttft_s,
                        total_latency_s=response.total_latency_s,
                        decode_tps=decode_tps,
                        base_url=merged["base_url"],
                        region=merged.get("region", ""),
                        gpu_type=merged.get("gpu_type", ""),
                        gpu_count=merged.get("gpu_count", 1),
                        tp_size=merged.get("tp_size", 1),
                    )
                    f.write(json.dumps(row.to_dict()) + "\n")
    finally:
        if active_server is not None and not keep_server:
            active_server.stop()

    return output
