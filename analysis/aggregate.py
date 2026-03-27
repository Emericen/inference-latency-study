from __future__ import annotations

from pathlib import Path

import pandas as pd


def aggregate_jsonl(input_path: str, output_path: str) -> Path:
    df = pd.read_json(input_path, lines=True)
    df = df[df["is_warmup"] == False].copy()

    grouped = (
        df.groupby(
            [
                "experiment_name",
                "scenario_id",
                "case_id",
                "model_id",
                "arch_type",
                "precision",
                "server_mode",
                "client_mode",
                "payload_kind",
                "cache_mode",
                "prompt_tokens",
                "image_count",
                "image_bytes",
                "base_url",
                "region",
                "gpu_type",
                "gpu_count",
                "tp_size",
            ],
            dropna=False,
        )
        .agg(
            requests=("request_index", "count"),
            ttft_p50_s=("ttft_s", lambda s: s.quantile(0.5)),
            ttft_p95_s=("ttft_s", lambda s: s.quantile(0.95)),
            ttft_mean_s=("ttft_s", "mean"),
            total_p50_s=("total_latency_s", lambda s: s.quantile(0.5)),
            total_p95_s=("total_latency_s", lambda s: s.quantile(0.95)),
            total_mean_s=("total_latency_s", "mean"),
            decode_tps_p50=("decode_tps", lambda s: s.quantile(0.5)),
            decode_tps_mean=("decode_tps", "mean"),
            completion_tokens_mean=("completion_tokens", "mean"),
        )
        .reset_index()
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out, index=False)
    return out
