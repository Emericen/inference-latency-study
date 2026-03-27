from __future__ import annotations

import base64
import io
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

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


@dataclass(frozen=True)
class PreparedImageRecord:
    bucket_bytes: int
    index: int
    path: str
    image_bytes: int
    image_width: int
    image_height: int
    source_width: int
    source_height: int


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
    best_image = current

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
                best_image = current
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
        image=best_image,
        encoded_bytes=best_bytes,
        mime_type="image/jpeg",
    )


def _rank_sample_indices(samples: list[ScreenSpotSample], target_image_bytes: int) -> list[int]:
    desired_min_bytes = int(target_image_bytes * 1.5)
    return sorted(
        range(len(samples)),
        key=lambda idx: (
            0 if len(samples[idx].image_bytes) >= desired_min_bytes else 1,
            abs(len(samples[idx].image_bytes) - desired_min_bytes),
            idx,
        ),
    )


def prepare_bucketed_images(
    *,
    repo_id: str,
    parquet_path: str,
    output_dir: Path,
    manifest_path: Path,
    bucket_targets: list[int],
    per_bucket: int,
    revision: str | None = None,
    tolerance_ratio: float = 0.10,
) -> Path:
    samples = load_screenspot_samples(
        repo_id=repo_id,
        parquet_path=parquet_path,
        revision=revision,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    prepared: dict[str, list[dict]] = {}
    for bucket_bytes in bucket_targets:
        ranked_indices = _rank_sample_indices(samples, bucket_bytes)
        selected: list[PreparedImageRecord] = []
        used_indices: set[int] = set()

        for idx in ranked_indices:
            if idx in used_indices:
                continue
            sample = samples[idx]
            encoded = _fit_image_to_target_bytes(
                _decode_sample_image(sample),
                target_bytes=bucket_bytes,
                tolerance_ratio=tolerance_ratio,
            )
            gap = abs(len(encoded.encoded_bytes) - bucket_bytes)
            if gap > int(bucket_bytes * tolerance_ratio):
                continue

            entry_index = len(selected)
            file_name = f"bucket_{bucket_bytes}_{entry_index:02d}.jpg"
            file_path = output_dir / file_name
            file_path.write_bytes(encoded.encoded_bytes)
            record = PreparedImageRecord(
                bucket_bytes=bucket_bytes,
                index=entry_index,
                path=file_name,
                image_bytes=len(encoded.encoded_bytes),
                image_width=encoded.image.width,
                image_height=encoded.image.height,
                source_width=sample.image_width,
                source_height=sample.image_height,
            )
            selected.append(record)
            used_indices.add(idx)
            if len(selected) >= per_bucket:
                break

        if len(selected) < per_bucket:
            raise ValueError(
                f"Only prepared {len(selected)} images for bucket {bucket_bytes}, expected {per_bucket}"
            )
        prepared[str(bucket_bytes)] = [asdict(record) for record in selected]

    manifest = {
        "repo_id": repo_id,
        "parquet_path": parquet_path,
        "revision": revision,
        "bucket_targets": bucket_targets,
        "per_bucket": per_bucket,
        "records": prepared,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


@lru_cache(maxsize=16)
def load_prepared_manifest(manifest_path: str) -> dict[str, list[PreparedImageRecord]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = manifest["records"]
    return {
        bucket: [PreparedImageRecord(**record) for record in bucket_records]
        for bucket, bucket_records in records.items()
    }


def load_prepared_encoded_image(
    *,
    manifest_path: str,
    target_image_bytes: int,
    seed: int,
) -> EncodedImage:
    manifest_records = load_prepared_manifest(manifest_path)
    bucket_key = str(target_image_bytes)
    if bucket_key not in manifest_records:
        raise ValueError(
            f"Bucket {target_image_bytes} not found in prepared manifest: {manifest_path}"
        )
    records = manifest_records[bucket_key]
    record = records[seed % len(records)]
    manifest_dir = Path(manifest_path).parent
    image_path = manifest_dir / record.path
    encoded_bytes = image_path.read_bytes()
    image = Image.open(io.BytesIO(encoded_bytes)).convert("RGB")
    return EncodedImage(
        image=image,
        encoded_bytes=encoded_bytes,
        mime_type="image/jpeg",
    )


def encoded_image_to_data_url(encoded_image: EncodedImage) -> tuple[str, int]:
    raw = encoded_image.encoded_bytes
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{encoded_image.mime_type};base64,{encoded}", len(raw)
