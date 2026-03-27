from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from data.screenspot import (
    encoded_image_to_data_url,
    load_prepared_encoded_image,
)
from data.text import build_text_for_token_budget
from data.token_budget import (
    count_image_prompt_tokens,
    count_multi_image_prompt_tokens,
    count_text_only_overhead_tokens,
    count_text_only_prompt_tokens,
)


@dataclass
class BuiltPayload:
    messages: list[dict]
    prompt_tokens: int
    vision_tokens_total: int
    image_count: int
    image_bytes: int


def _build_synthetic_image(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _image_to_data_url(image: Image.Image, fmt: str = "PNG") -> tuple[str, int]:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    raw = buf.getvalue()
    encoded = base64.b64encode(raw).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}", len(raw)


def build_payload(
    *,
    tokenizer,
    image_processor,
    payload_kind: str,
    target_prompt_tokens: int,
    question: str,
    seed: int,
    image_width: int,
    image_height: int,
    image_count: int = 1,
    screenspot_repo_id: str | None = None,
    screenspot_parquet_path: str | None = None,
    screenspot_revision: str | None = None,
    prepared_manifest_path: str | None = None,
    prepared_index: int | None = None,
    target_image_bytes: int | None = None,
) -> BuiltPayload:
    if payload_kind == "text_only":
        overhead = count_text_only_overhead_tokens(
            tokenizer=tokenizer, question=question
        )
        body_budget = max(1, target_prompt_tokens - overhead)
        prompt_tokens = None
        text = ""
        adjusted_budget = body_budget
        while adjusted_budget > 0:
            text = build_text_for_token_budget(
                tokenizer=tokenizer,
                target_tokens=adjusted_budget,
                seed=seed,
            )
            prompt_tokens = count_text_only_prompt_tokens(
                tokenizer=tokenizer,
                text=text,
                question=question,
            )
            if prompt_tokens <= target_prompt_tokens:
                break
            adjusted_budget -= 1

        messages = [{"role": "user", "content": f"{text}\n\n --- \n\n{question}"}]
        return BuiltPayload(
            messages=messages,
            prompt_tokens=prompt_tokens,
            vision_tokens_total=0,
            image_count=0,
            image_bytes=0,
        )

    if payload_kind == "image_text":
        images = [
            _build_synthetic_image(
                width=image_width,
                height=image_height,
                seed=seed + image_index,
            )
            for image_index in range(image_count)
        ]
        if image_count == 1:
            prompt_tokens, image_token_count = count_image_prompt_tokens(
                tokenizer=tokenizer,
                image_processor=image_processor,
                image=images[0],
                question=question,
            )
            vision_tokens_total = image_token_count
        else:
            prompt_tokens, image_token_counts = count_multi_image_prompt_tokens(
                tokenizer=tokenizer,
                image_processor=image_processor,
                images=images,
                question=question,
            )
            vision_tokens_total = sum(image_token_counts)

        content = []
        image_bytes = 0
        for image in images:
            data_url, current_image_bytes = _image_to_data_url(image)
            image_bytes += current_image_bytes
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        content.append({"type": "text", "text": question})
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        return BuiltPayload(
            messages=messages,
            prompt_tokens=prompt_tokens,
            vision_tokens_total=vision_tokens_total,
            image_count=image_count,
            image_bytes=image_bytes,
        )

    if payload_kind == "prepared_image_text":
        if prepared_manifest_path is None:
            raise ValueError("prepared_image_text payloads require prepared_manifest_path")
        if prepared_index is None:
            raise ValueError("prepared_image_text payloads require prepared_index")
        if target_image_bytes is None:
            raise ValueError(
                "prepared_image_text payloads require target_image_bytes"
            )
        image_count = max(1, image_count)
        encoded_images = [
            load_prepared_encoded_image(
                manifest_path=prepared_manifest_path,
                target_image_bytes=target_image_bytes,
                prepared_index=prepared_index + image_index,
            )
            for image_index in range(image_count)
        ]
        images = [encoded_image.image for encoded_image in encoded_images]
        if image_count == 1:
            prompt_tokens, image_token_count = count_image_prompt_tokens(
                tokenizer=tokenizer,
                image_processor=image_processor,
                image=images[0],
                question=question,
            )
            vision_tokens_total = image_token_count
        else:
            prompt_tokens, image_token_counts = count_multi_image_prompt_tokens(
                tokenizer=tokenizer,
                image_processor=image_processor,
                images=images,
                question=question,
            )
            vision_tokens_total = sum(image_token_counts)

        content = []
        image_bytes = 0
        for encoded_image in encoded_images:
            data_url, current_image_bytes = encoded_image_to_data_url(encoded_image)
            image_bytes += current_image_bytes
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        return BuiltPayload(
            messages=messages,
            prompt_tokens=prompt_tokens,
            vision_tokens_total=vision_tokens_total,
            image_count=image_count,
            image_bytes=image_bytes,
        )

    raise ValueError(f"Unsupported payload_kind: {payload_kind}")
