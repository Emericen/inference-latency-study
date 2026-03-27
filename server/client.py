from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class RequestMetrics:
    content: str
    ttft_s: float | None
    total_latency_s: float


class VLLMClient:
    def __init__(self, *, base_url: str, api_key: str = "EMPTY", timeout_s: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=timeout_s,
        )

    def run_request(
        self,
        *,
        model: str,
        messages: list[dict],
        max_completion_tokens: int,
    ) -> RequestMetrics:
        start = time.perf_counter()
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_completion_tokens,
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

        total_latency_s = time.perf_counter() - start
        return RequestMetrics(
            content="".join(content_parts),
            ttft_s=ttft_s,
            total_latency_s=total_latency_s,
        )
