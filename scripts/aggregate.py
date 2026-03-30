"""Aggregate raw JSONL benchmark results into a CSV summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aggregate_jsonl(input_path: str, output_path: str) -> Path:
    df = pd.read_json(input_path, lines=True)
    df = df[df["is_warmup"] == False].copy()

    grouped = (
        df.groupby(["modality", "history_images", "model", "base_url"], dropna=False)
        .agg(
            requests=("row_index", "count"),
            vision_tokens_mean=("vision_tokens", "mean"),
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
    parser = argparse.ArgumentParser(description="Aggregate raw JSONL into CSV summary.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = aggregate_jsonl(input_path=args.input, output_path=args.output)
    print(out)


if __name__ == "__main__":
    main()
