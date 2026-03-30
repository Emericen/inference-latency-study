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
    history_images: int
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


def _build_messages(rows: list[dict], modality: str, question: str) -> list[dict]:
    if modality == "image":
        content = [
            {"type": "image_url", "image_url": {"url": _image_to_data_url(row["image"])}}
            for row in rows
        ]
        content.append({"type": "text", "text": question})
        return [{"role": "user", "content": content}]
    else:
        row = rows[0]
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
    history_images: int = 1,
    row_offset: int = 0,
    history_mode: str = "sliding",
    api_key: str = "EMPTY",
) -> Path:
    ds = load_from_disk(dataset_path)
    total_needed = warmups + runs
    if modality == "image":
        if history_mode == "sliding":
            required_rows = row_offset + total_needed + max(0, history_images - 1)
        elif history_mode == "disjoint":
            required_rows = row_offset + (total_needed * history_images)
        elif history_mode == "prefix":
            required_rows = row_offset + history_images
        else:
            raise ValueError(f"Unsupported history_mode={history_mode!r}")
    else:
        required_rows = row_offset + total_needed
    if len(ds) < required_rows:
        raise ValueError(
            f"Dataset has {len(ds)} rows but need {required_rows} "
            f"(row_offset={row_offset}, warmups={warmups}, runs={runs}, "
            f"history_images={history_images}, history_mode={history_mode})"
        )

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300.0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for i in range(total_needed):
            is_warmup = i < warmups
            if modality == "image":
                if history_mode == "sliding":
                    start = row_offset + i
                    rows = [ds[start + offset] for offset in range(history_images)]
                elif history_mode == "disjoint":
                    start = row_offset + (i * history_images)
                    rows = [ds[start + offset] for offset in range(history_images)]
                else:
                    rows = [ds[row_offset + offset] for offset in range(history_images)]
            else:
                rows = [ds[row_offset + i]]
            messages = _build_messages(rows, modality, question)
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
                history_images=history_images if modality == "image" else 0,
                is_warmup=is_warmup,
                model=model,
                vision_tokens=sum(row["vision_tokens"] for row in rows) if modality == "image" else rows[0]["vision_tokens"],
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
    parser.add_argument("--model", default="MODEL")
    parser.add_argument("--modality", required=True, choices=["image", "text"])
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--question", default="What do you see?")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--history-images", type=int, default=1, help="Number of images to include per request when modality=image.")
    parser.add_argument("--history-mode", choices=["sliding", "disjoint", "prefix"], default="sliding", help="How image history windows are constructed.")
    parser.add_argument("--row-offset", type=int, default=0, help="Starting dataset row offset for this run.")
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
        history_images=args.history_images,
        row_offset=args.row_offset,
        history_mode=args.history_mode,
        api_key=args.api_key,
    )
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
