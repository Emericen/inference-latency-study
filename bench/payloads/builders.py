from __future__ import annotations

from dataclasses import dataclass

from bench.payloads.synthetic_image import build_synthetic_image, image_to_data_url
from bench.payloads.synthetic_text import build_text_for_token_budget
from bench.token_budget import (
    count_image_prompt_tokens,
    count_text_only_overhead_tokens,
    count_text_only_prompt_tokens,
)


@dataclass
class BuiltPayload:
    messages: list[dict]
    prompt_tokens: int
    image_count: int
    image_bytes: int


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
            image_count=0,
            image_bytes=0,
        )

    if payload_kind == "image_text":
        image = build_synthetic_image(width=image_width, height=image_height, seed=seed)
        data_url, image_bytes = image_to_data_url(image)
        prompt_tokens, _ = count_image_prompt_tokens(
            tokenizer=tokenizer,
            image_processor=image_processor,
            image=image,
            question=question,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return BuiltPayload(
            messages=messages,
            prompt_tokens=prompt_tokens,
            image_count=1,
            image_bytes=image_bytes,
        )

    raise ValueError(f"Unsupported payload_kind: {payload_kind}")
