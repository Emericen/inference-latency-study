"""Run the latency benchmark against a vLLM server.

Loads the pre-built dataset, iterates rows, and measures TTFT / total
latency for either image+text or text-only modality.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import load_from_disk
from openai import OpenAI

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "prepared" / "benchmark_dataset"


@dataclass
class ResultRow:
    row_index: int
    modality: str
    is_warmup: bool
    model: str
    vision_tokens: int
    ttft_s: float | None
    total_latency_s: float
    completion_tokens: int
    decode_tps: float | None
    base_url: str


def _image_to_data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _build_messages(row: dict, modality: str, question: str) -> list[dict]:
    if modality == "image":
        return [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _image_to_data_url(row["image"])}},
            {"type": "text", "text": question},
        ]}]
    else:
        return [{"role": "user", "content": f"{row['text']}\n\n --- \n\n{question}"}]


def _send_request(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> tuple[str, float | None, float]:
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    )

    ttft_s = None
    parts: list[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            if ttft_s is None:
                ttft_s = time.perf_counter() - start
            parts.append(content)

    total_s = time.perf_counter() - start
    return "".join(parts), ttft_s, total_s


def run(
    *,
    base_url: str,
    model: str,
    modality: str,
    dataset_path: str,
    output_path: str,
    question: str = "What do you see?",
    max_tokens: int = 10,
    warmups: int = 2,
    runs: int = 10,
    api_key: str = "EMPTY",
) -> Path:
    ds = load_from_disk(dataset_path)
    total_needed = warmups + runs
    if len(ds) < total_needed:
        raise ValueError(f"Dataset has {len(ds)} rows but need {total_needed} (warmups={warmups}, runs={runs})")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300.0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for i in range(total_needed):
            is_warmup = i < warmups
            row = ds[i]
            messages = _build_messages(row, modality, question)
            content, ttft_s, total_s = _send_request(client, model, messages, max_tokens)

            # rough token count from whitespace split (avoids loading tokenizer)
            completion_tokens = max(1, len(content.split()))
            decode_tps = None
            if ttft_s is not None and completion_tokens > 1:
                decode_window = total_s - ttft_s
                if decode_window > 0:
                    decode_tps = (completion_tokens - 1) / decode_window

            result = ResultRow(
                row_index=i,
                modality=modality,
                is_warmup=is_warmup,
                model=model,
                vision_tokens=row["vision_tokens"],
                ttft_s=ttft_s,
                total_latency_s=total_s,
                completion_tokens=completion_tokens,
                decode_tps=decode_tps,
                base_url=base_url,
            )
            f.write(json.dumps(asdict(result)) + "\n")
            f.flush()

            tag = "warmup" if is_warmup else f"run {i - warmups + 1}"
            print(f"[{tag}] row={i} ttft={ttft_s:.3f}s total={total_s:.3f}s" if ttft_s else f"[{tag}] row={i} total={total_s:.3f}s")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference latency benchmark.")
    parser.add_argument("--base-url", required=True, help="vLLM server URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--modality", required=True, choices=["image", "text"])
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--question", default="What do you see?")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--api-key", default="EMPTY")
    args = parser.parse_args()

    out = run(
        base_url=args.base_url,
        model=args.model,
        modality=args.modality,
        dataset_path=args.dataset,
        output_path=args.output,
        question=args.question,
        max_tokens=args.max_tokens,
        warmups=args.warmups,
        runs=args.runs,
        api_key=args.api_key,
    )
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
