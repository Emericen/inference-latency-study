from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
                "image_count",
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
            prompt_tokens_mean=("prompt_tokens", "mean"),
            vision_tokens_total_mean=("vision_tokens_total", "mean"),
            image_bytes_mean=("image_bytes", "mean"),
            request_bytes_total_mean=("request_bytes_total", "mean"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate raw latency benchmark JSONL into CSV.")
    parser.add_argument("--input", required=True, help="Path to raw JSONL results.")
    parser.add_argument("--output", required=True, help="Path to summary CSV output.")
    args = parser.parse_args()
    out = aggregate_jsonl(input_path=args.input, output_path=args.output)
    print(out)


if __name__ == "__main__":
    main()
