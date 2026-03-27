from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def build_synthetic_image(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> tuple[str, int]:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    raw = buf.getvalue()
    encoded = base64.b64encode(raw).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}", len(raw)
