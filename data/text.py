from __future__ import annotations

import random
import string


def _random_word(rng: random.Random) -> str:
    length = rng.randint(3, 9)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def _fill_exact_gap(tokenizer, text: str, target_tokens: int) -> str:
    candidates = [" a", " the", " x", ".", ",", "\n", " 0", " foo", " bar"]
    current = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if current == target_tokens:
        return text
    if current > target_tokens:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:target_tokens]
        return tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    gap = target_tokens - current
    for candidate in candidates:
        candidate_tokens = len(
            tokenizer(candidate, add_special_tokens=False)["input_ids"]
        )
        if candidate_tokens <= gap:
            attempted = text + candidate
            attempted_count = len(
                tokenizer(attempted, add_special_tokens=False)["input_ids"]
            )
            if attempted_count <= target_tokens:
                return _fill_exact_gap(tokenizer, attempted, target_tokens)
    return text


def build_text_for_token_budget(tokenizer, target_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    words: list[str] = []

    while True:
        words.append(_random_word(rng))
        candidate = " ".join(words)
        count = len(tokenizer(candidate, add_special_tokens=False)["input_ids"])
        if count >= target_tokens:
            break

    ids = tokenizer(candidate, add_special_tokens=False)["input_ids"][:target_tokens]
    text = tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    text = _fill_exact_gap(tokenizer, text, target_tokens)

    final_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(final_ids) > target_tokens:
        text = tokenizer.decode(
            final_ids[:target_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return text
