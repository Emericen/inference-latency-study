"""Run screenshot-based VLM latency studies against a vLLM server."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from openai import OpenAI
from PIL import Image

from aggregate import aggregate_jsonl

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS_DIR = ROOT / "data" / "screenshots"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


# fmt: off
@dataclass
class RunConfig:
    experiment: str  # Experiment name used in outputs, e.g. "dense_scale_qwen3_vl_8b_local".
    base_url: str  # vLLM or provider base URL, e.g. "http://localhost:8000/v1".
    model: str  # Model name sent to the API, e.g. "Qwen/Qwen3-VL-8B-Instruct".
    region: str = "unknown"  # Human-readable server location tag, e.g. "local" or "remote-near".
    question: str = "What do you see?"  # Text block repeated for each request unit.
    max_tokens: int = 10  # Maximum completion tokens to request from the model.
    warmup_size: int = 10  # Number of warmup requests before the measured context sweep.
    context_mode: str = "full_history"  # "full_history" keeps all past screenshots; "omit_past_history" replaces prior screenshots with placeholder text.
    context_max_size: int = 100  # Run a measured sweep from context size 1 through this maximum on one live server.
    screenshots_dir: str = str(SCREENSHOTS_DIR)  # Directory containing the ordered screenshot corpus.
    output_path: str = "results/raw/run.jsonl"  # Destination JSONL for raw per-request results.
    api_key: str = "EMPTY"  # API key for the target endpoint; vLLM typically accepts "EMPTY".
    image_placeholder: str = "[image omitted]"  # Text used when omitting prior images in omit-past mode.
    

@dataclass
class ResultRow:
    experiment: str
    row_index: int
    is_warmup: bool
    model: str
    region: str
    context_mode: str
    context_size: int
    request_bytes: int
    ttft_s: float | None
    total_latency_s: float
    completion_tokens: int
    base_url: str
# fmt: on


def _load_config(config_path: str) -> RunConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RunConfig(**raw)


def _load_screenshots(screenshots_dir: str | Path) -> list[Path]:
    paths = sorted(Path(screenshots_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg files found in {screenshots_dir}")
    return paths


def _image_to_bytes(path: Path) -> bytes:
    with Image.open(path).convert("RGB") as image:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def _image_to_data_url(path: Path) -> tuple[str, int]:
    raw = _image_to_bytes(path)
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}", len(raw)


def _build_window(paths: list[Path], *, context_size: int) -> list[Path]:
    start = 0
    end = start + context_size
    if end > len(paths):
        raise ValueError(
            f"Need rows up to {end}, but only {len(paths)} screenshots are available."
        )
    return paths[start:end]


def _build_messages(
    image_paths: list[Path],
    *,
    context_mode: str,
    question: str,
    image_placeholder: str,
) -> tuple[list[dict], int]:
    if context_mode not in {"full_history", "omit_past_history"}:
        raise ValueError(f"Unsupported context_mode={context_mode!r}")

    content: list[dict] = []
    request_bytes = 0

    for i, path in enumerate(image_paths):
        content.append({"type": "text", "text": question})
        request_bytes += len(question.encode("utf-8"))

        if context_mode == "full_history" or i == len(image_paths) - 1:
            data_url, image_bytes = _image_to_data_url(path)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
            request_bytes += image_bytes
        else:
            content.append({"type": "text", "text": image_placeholder})
            request_bytes += len(image_placeholder.encode("utf-8"))

    return [{"role": "user", "content": content}], request_bytes


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


def run(config: RunConfig) -> tuple[Path, Path]:
    screenshots = _load_screenshots(config.screenshots_dir)
    context_sizes = list(range(1, config.context_max_size + 1))
    total_needed = context_sizes[-1]
    if len(screenshots) < total_needed:
        raise ValueError(
            f"Need at least {total_needed} screenshots, found {len(screenshots)}."
        )

    client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=300.0)
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for i in range(config.warmup_size):
            image_paths = _build_window(screenshots, context_size=1)
            messages, _ = _build_messages(
                image_paths,
                context_mode=config.context_mode,
                question=config.question,
                image_placeholder=config.image_placeholder,
            )
            _send_request(client, config.model, messages, config.max_tokens)

        for i, context_size in enumerate(context_sizes):
            image_paths = _build_window(screenshots, context_size=context_size)
            messages, request_bytes = _build_messages(
                image_paths,
                context_mode=config.context_mode,
                question=config.question,
                image_placeholder=config.image_placeholder,
            )
            content, ttft_s, total_s = _send_request(
                client, config.model, messages, config.max_tokens
            )
            completion_tokens = max(1, len(content.split()))

            result = ResultRow(
                experiment=config.experiment,
                row_index=i,
                is_warmup=False,
                model=config.model,
                region=config.region,
                context_mode=config.context_mode,
                context_size=context_size,
                request_bytes=request_bytes,
                ttft_s=ttft_s,
                total_latency_s=total_s,
                completion_tokens=completion_tokens,
                base_url=config.base_url,
            )
            f.write(json.dumps(asdict(result)) + "\n")
            f.flush()

            tag = f"n={context_size}"
            if ttft_s is None:
                print(f"[{tag}] row={i} total={total_s:.3f}s")
            else:
                print(f"[{tag}] row={i} ttft={ttft_s:.3f}s total={total_s:.3f}s")

    summary = aggregate_jsonl(str(output))
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run screenshot-based VLM latency study."
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--base-url", help="Override base_url from config")
    parser.add_argument("--output", help="Override output_path from config")
    parser.add_argument("--model", help="Override model from config")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.base_url:
        config.base_url = args.base_url
    if args.output:
        config.output_path = args.output
    if args.model:
        config.model = args.model

    raw_path, summary_path = run(config)
    print(f"Raw results written to {raw_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
