from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ResultRow:
    experiment_name: str
    scenario_id: str
    case_id: str
    request_index: int
    is_warmup: bool
    model_id: str
    arch_type: str
    precision: str
    server_mode: str
    client_mode: str
    payload_kind: str
    cache_mode: str
    prompt_tokens: int
    completion_tokens: int
    image_count: int
    image_bytes: int
    ttft_s: float | None
    total_latency_s: float
    decode_tps: float | None
    base_url: str
    region: str
    gpu_type: str
    gpu_count: int
    tp_size: int

    def to_dict(self) -> dict:
        return asdict(self)
