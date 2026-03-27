from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from functools import lru_cache

from huggingface_hub import hf_hub_download
from PIL import Image
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ScreenSpotSample:
    image_bytes: bytes
    image_width: int
    image_height: int


@dataclass(frozen=True)
class EncodedImage:
    image: Image.Image
    encoded_bytes: bytes
    mime_type: str


def _load_parquet_rows(parquet_path: str) -> list[ScreenSpotSample]:
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    samples: list[ScreenSpotSample] = []
    for row in rows:
        image_value = row.get("image")
        if not isinstance(image_value, dict):
            continue
        image_bytes = image_value.get("bytes")
        image_width = row.get("image_width")
        image_height = row.get("image_height")
        if not image_bytes or image_width is None or image_height is None:
            continue
        samples.append(
            ScreenSpotSample(
                image_bytes=image_bytes,
                image_width=int(image_width),
                image_height=int(image_height),
            )
        )
    if not samples:
        raise ValueError(f"No image samples found in parquet file: {parquet_path}")
    return samples


@lru_cache(maxsize=8)
def load_screenspot_samples(
    *,
    repo_id: str,
    parquet_path: str,
    revision: str | None = None,
) -> list[ScreenSpotSample]:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=parquet_path,
        repo_type="dataset",
        revision=revision,
    )
    return _load_parquet_rows(local_path)


def _decode_sample_image(sample: ScreenSpotSample) -> Image.Image:
    image = Image.open(io.BytesIO(sample.image_bytes))
    return image.convert("RGB")


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _fit_image_to_target_bytes(
    image: Image.Image,
    *,
    target_bytes: int,
    tolerance_ratio: float = 0.05,
    min_quality: int = 20,
    max_quality: int = 95,
    min_scale: float = 0.4,
) -> EncodedImage:
    current = image
    scale = 1.0
    best_bytes: bytes | None = None
    best_gap: int | None = None

    while scale >= min_scale:
        low = min_quality
        high = max_quality
        candidate_bytes: bytes | None = None
        candidate_gap: int | None = None

        while low <= high:
            quality = (low + high) // 2
            encoded = _encode_jpeg(current, quality=quality)
            size = len(encoded)
            gap = abs(size - target_bytes)
            if candidate_gap is None or gap < candidate_gap:
                candidate_bytes = encoded
                candidate_gap = gap
            if size > target_bytes:
                high = quality - 1
            else:
                low = quality + 1

        if candidate_bytes is not None:
            candidate_size = len(candidate_bytes)
            gap = abs(candidate_size - target_bytes)
            if best_gap is None or gap < best_gap:
                best_bytes = candidate_bytes
                best_gap = gap
            if gap <= int(target_bytes * tolerance_ratio):
                return EncodedImage(
                    image=current,
                    encoded_bytes=candidate_bytes,
                    mime_type="image/jpeg",
                )

        scale *= 0.9
        next_width = max(64, int(round(image.width * scale)))
        next_height = max(64, int(round(image.height * scale)))
        current = image.resize((next_width, next_height), Image.Resampling.LANCZOS)

    if best_bytes is None:
        raise ValueError("Unable to encode image to target byte budget")
    return EncodedImage(
        image=current,
        encoded_bytes=best_bytes,
        mime_type="image/jpeg",
    )


def build_screenspot_encoded_image(
    *,
    repo_id: str,
    parquet_path: str,
    target_image_bytes: int,
    seed: int,
    revision: str | None = None,
    search_limit: int = 64,
    tolerance_ratio: float = 0.05,
) -> EncodedImage:
    samples = load_screenspot_samples(
        repo_id=repo_id,
        parquet_path=parquet_path,
        revision=revision,
    )
    desired_min_bytes = int(target_image_bytes * 1.5)
    ranked_indices = sorted(
        range(len(samples)),
        key=lambda idx: (
            0 if len(samples[idx].image_bytes) >= desired_min_bytes else 1,
            abs(len(samples[idx].image_bytes) - desired_min_bytes),
            (idx - seed) % len(samples),
        ),
    )
    best_image: EncodedImage | None = None
    best_gap: int | None = None

    for idx in ranked_indices[: min(search_limit, len(samples))]:
        sample = samples[idx]
        image = _decode_sample_image(sample)
        encoded = _fit_image_to_target_bytes(
            image,
            target_bytes=target_image_bytes,
            tolerance_ratio=tolerance_ratio,
        )
        gap = abs(len(encoded.encoded_bytes) - target_image_bytes)
        if best_gap is None or gap < best_gap:
            best_image = encoded
            best_gap = gap
        if gap <= int(target_image_bytes * tolerance_ratio):
            return encoded

    if best_image is None:
        raise ValueError("Unable to find ScreenSpot sample for target byte bucket")
    return best_image


def encoded_image_to_data_url(encoded_image: EncodedImage) -> tuple[str, int]:
    raw = encoded_image.encoded_bytes
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{encoded_image.mime_type};base64,{encoded}", len(raw)
