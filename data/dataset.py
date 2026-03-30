"""Build and load the benchmark dataset.

Each row pairs a screenshot with a token-matched random text chunk.
The runner picks a row and assembles either an image+text or text-only
message depending on the modality being tested.

Columns:
    image       — PIL Image (the screenshot)
    text        — random tokens matching that image's vision token count
    vision_tokens — shared token count for the pair
"""

from __future__ import annotations

import random
import string
from pathlib import Path

from datasets import Dataset, Features, Image, Value, load_from_disk
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoTokenizer

SCREENSHOTS_DIR = Path(__file__).resolve().parent / "screenshots"
DATASET_DIR = Path(__file__).resolve().parent / "prepared" / "benchmark_dataset"

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


# --- token counting --------------------------------------------------------


def _load_tokenizer(model: str, local_files_only: bool = True):
    return AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)


def _load_image_processor(model: str, local_files_only: bool = True):
    return AutoImageProcessor.from_pretrained(model, local_files_only=local_files_only)


def _count_vision_tokens(image: PILImage.Image, image_processor) -> int:
    inputs = image_processor(images=[image])
    grid = inputs["image_grid_thw"][0]
    return int(grid.prod() // (image_processor.merge_size ** 2))


# --- random text generation ------------------------------------------------


def _random_word(rng: random.Random) -> str:
    length = rng.randint(3, 9)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def _build_text_for_tokens(tokenizer, target_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    words: list[str] = []

    while True:
        words.append(_random_word(rng))
        candidate = " ".join(words)
        count = len(tokenizer(candidate, add_special_tokens=False)["input_ids"])
        if count >= target_tokens:
            break

    ids = tokenizer(candidate, add_special_tokens=False)["input_ids"][:target_tokens]
    return tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )


# --- dataset build ----------------------------------------------------------


def build_dataset(
    *,
    screenshots_dir: str | Path = SCREENSHOTS_DIR,
    output_dir: str | Path = DATASET_DIR,
    model: str = DEFAULT_MODEL,
    local_files_only: bool = True,
    seed: int = 42,
) -> Dataset:
    """Scan screenshots, count vision tokens, generate matched text, save."""
    screenshots_dir = Path(screenshots_dir)
    output_dir = Path(output_dir)

    paths = sorted(screenshots_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg files found in {screenshots_dir}")

    tokenizer = _load_tokenizer(model, local_files_only=local_files_only)
    image_processor = _load_image_processor(model, local_files_only=local_files_only)

    images = []
    texts = []
    vision_tokens_list = []

    for i, path in enumerate(paths):
        img = PILImage.open(path).convert("RGB")
        vt = _count_vision_tokens(img, image_processor)
        text = _build_text_for_tokens(tokenizer, target_tokens=vt, seed=seed + i)
        images.append(img)
        texts.append(text)
        vision_tokens_list.append(vt)

    ds = Dataset.from_dict(
        {"image": images, "text": texts, "vision_tokens": vision_tokens_list},
        features=Features({
            "image": Image(),
            "text": Value("string"),
            "vision_tokens": Value("int32"),
        }),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print(f"Saved {len(ds)} rows to {output_dir}")
    return ds


def load_dataset(path: str | Path = DATASET_DIR) -> Dataset:
    """Load a previously built benchmark dataset."""
    return load_from_disk(str(path))


# --- CLI entry point --------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build benchmark dataset from screenshots.")
    parser.add_argument("--screenshots-dir", default=str(SCREENSHOTS_DIR))
    parser.add_argument("--output-dir", default=str(DATASET_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_dataset(
        screenshots_dir=args.screenshots_dir,
        output_dir=args.output_dir,
        model=args.model,
        local_files_only=not args.no_local_files_only,
        seed=args.seed,
    )
