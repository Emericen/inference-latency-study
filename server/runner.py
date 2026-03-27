from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from data.payloads import build_payload
from data.token_budget import load_token_tools
from server.client import VLLMClient
from server.schema import ResultRow
from server.vllm import wait_until_ready


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _estimate_request_bytes_total(
    *,
    model: str,
    messages: list[dict],
    max_completion_tokens: int,
) -> int:
    request_body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_completion_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    return len(json.dumps(request_body, separators=(",", ":")).encode("utf-8"))


def _build_client(case: dict) -> VLLMClient:
    api_key = case.get("api_key")
    if api_key is None and case.get("api_key_env"):
        api_key = os.environ.get(case["api_key_env"])
    if not api_key:
        api_key = "EMPTY"
    return VLLMClient(
        base_url=case["base_url"],
        api_key=api_key,
        timeout_s=case.get("timeout_s", 300.0),
    )


def run_config_file(
    path: str,
    *,
    output_path: str | None = None,
    overrides: dict | None = None,
    wait_for_server_timeout_s: float = 30.0,
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

    if not cases:
        raise ValueError(f"No cases found in config: {config_path}")

    if wait_for_server_timeout_s > 0:
        probe_case = _deep_merge(defaults, cases[0])
        wait_until_ready(probe_case["base_url"], timeout_s=wait_for_server_timeout_s)

    with output.open("w", encoding="utf-8") as f:
        for case in cases:
            merged = _deep_merge(defaults, case)
            client = _build_client(merged)
            token_tools = load_token_tools(
                merged["token_budget_model"],
                local_files_only=merged.get("local_files_only", True),
            )

            total_runs = merged["warmups"] + merged["runs"]
            prepared_index_offset = merged["payload"].get("prepared_index_offset", 0)
            for request_index in range(total_runs):
                is_warmup = request_index < merged["warmups"]
                seed = (
                    merged["seed"]
                    if merged["payload"]["cache_mode"] == "repeat"
                    else merged["seed"] + request_index
                )
                prepared_index = None
                if merged["payload"]["kind"] == "prepared_image_text":
                    if merged["payload"]["cache_mode"] == "repeat":
                        prepared_index = prepared_index_offset
                    else:
                        prepared_index = prepared_index_offset + request_index
                payload = build_payload(
                    tokenizer=token_tools.tokenizer,
                    image_processor=token_tools.image_processor,
                    payload_kind=merged["payload"]["kind"],
                    target_prompt_tokens=merged["payload"]["target_prompt_tokens"],
                    question=merged["payload"]["question"],
                    seed=seed,
                    image_width=merged["payload"].get("image_width", 1920),
                    image_height=merged["payload"].get("image_height", 1080),
                    image_count=merged["payload"].get("image_count", 1),
                    screenspot_repo_id=merged["payload"].get("screenspot_repo_id"),
                    screenspot_parquet_path=merged["payload"].get("screenspot_parquet_path"),
                    screenspot_revision=merged["payload"].get("screenspot_revision"),
                    prepared_manifest_path=merged["payload"].get("prepared_manifest_path"),
                    prepared_index=prepared_index,
                    target_image_bytes=merged["payload"].get("target_image_bytes"),
                )

                request_bytes_total = _estimate_request_bytes_total(
                    model=merged["model"],
                    messages=payload.messages,
                    max_completion_tokens=merged["max_completion_tokens"],
                )

                response = client.run_request(
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
                    vision_tokens_total=payload.vision_tokens_total,
                    completion_tokens=completion_tokens,
                    image_count=payload.image_count,
                    image_bytes=payload.image_bytes,
                    request_bytes_total=request_bytes_total,
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
                f.flush()

    return output
