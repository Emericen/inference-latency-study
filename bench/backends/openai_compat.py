from __future__ import annotations

import time

from openai import OpenAI

from bench.backends.base import BackendResponse, BenchmarkBackend


class OpenAICompatBackend(BenchmarkBackend):
    def __init__(self, *, base_url: str, api_key: str, timeout_s: float = 300.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

    def run_request(
        self,
        *,
        model: str,
        messages: list[dict],
        max_completion_tokens: int,
    ) -> BackendResponse:
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
        return BackendResponse(
            content="".join(content_parts),
            ttft_s=ttft_s,
            total_latency_s=total_latency_s,
        )
