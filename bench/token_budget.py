from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer


@dataclass
class TokenTools:
    tokenizer: object
    image_processor: object


@lru_cache(maxsize=8)
def load_token_tools(model_name: str, local_files_only: bool = True) -> TokenTools:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    return TokenTools(tokenizer=tokenizer, image_processor=image_processor)


def _text_only_template(text: str, question: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{text}\n\n --- \n\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _image_template(image_token_count: int, question: str) -> str:
    return (
        "<|im_start|>user\n"
        "<|vision_start|>"
        + ("<|image_pad|>" * image_token_count)
        + "<|vision_end|>"
        + question
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def count_text_only_prompt_tokens(
    tokenizer,
    text: str,
    question: str,
) -> int:
    prompt = _text_only_template(text=text, question=question)
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def count_text_only_overhead_tokens(tokenizer, question: str) -> int:
    return count_text_only_prompt_tokens(
        tokenizer=tokenizer, text="", question=question
    )


def count_image_prompt_tokens(
    tokenizer,
    image_processor,
    image: Image.Image,
    question: str,
) -> tuple[int, int]:
    image_inputs = image_processor(images=[image])
    image_grid_thw = image_inputs["image_grid_thw"][0]
    image_token_count = int(image_grid_thw.prod() // (image_processor.merge_size**2))
    prompt = _image_template(image_token_count=image_token_count, question=question)
    total_prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return total_prompt_tokens, image_token_count
