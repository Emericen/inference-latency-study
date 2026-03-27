from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI


def _stream_once(client: OpenAI, *, base_url: str, model: str, messages: list[dict]) -> None:
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        stream=True,
    )
    ttft_s = None
    content_parts: list[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            if ttft_s is None:
                ttft_s = time.perf_counter() - start
            content_parts.append(content)
    total_s = time.perf_counter() - start
    print(f"{base_url} ttft={ttft_s:.3f}s total={total_s:.3f}s text={''.join(content_parts)[:80]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a vLLM OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", required=True, help="Base URL such as https://<pod>-8000.proxy.runpod.net/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--image-path",
        default="data/prepared/screenspot_payload_buckets_quick/bucket_131072_00.jpg",
        help="Local image path for a multimodal smoke request.",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key="EMPTY", timeout=120)

    models = client.models.list()
    print(f"models={len(models.data)} first={models.data[0].id if models.data else 'none'}")

    text_messages = [{"role": "user", "content": "hello"}]
    _stream_once(client, base_url=args.base_url, model=args.model, messages=text_messages)

    image_bytes = Path(args.image_path).read_bytes()
    image_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")
    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]
    _stream_once(client, base_url=args.base_url, model=args.model, messages=image_messages)


if __name__ == "__main__":
    main()
