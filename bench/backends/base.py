from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BackendResponse:
    content: str
    ttft_s: float | None
    total_latency_s: float


class BenchmarkBackend(ABC):
    @abstractmethod
    def run_request(
        self,
        *,
        model: str,
        messages: list[dict],
        max_completion_tokens: int,
    ) -> BackendResponse:
        raise NotImplementedError
