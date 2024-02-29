from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from dataclasses_json import DataClassJsonMixin
from transformers import AutoTokenizer

from src import functional
from src.models import ModelandTokenizer
from src.utils.typing import Tokenizer


@contextmanager
def set_padding_side(
    tokenizer: AutoTokenizer, padding_side: str = "right"
) -> Iterator[None]:
    """Temporarily set padding side for tokenizer.

    Useful for when you want to batch generate with causal LMs like GPT, as these
    require the padding to be on the left side in such settings but are much easier
    to mess around with when the padding, by default, is on the right.

    Example usage:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        with tokenizer_utils.set_padding_side(tokenizer, "left"):
            inputs = mt.tokenizer(...)
        # ... later
        model.generate(**inputs)

    """
    _padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    yield
    tokenizer.padding_side = _padding_side


def maybe_prefix_eos(tokenizer: Tokenizer | ModelandTokenizer, prompt: str) -> str:
    """Prefix prompt with EOS token if model has no special start token."""
    tokenizer = functional.unwrap_tokenizer(tokenizer)
    if hasattr(tokenizer, "eos_token"):
        prefix = tokenizer.eos_token
        if not prompt.startswith(prefix):
            prompt = prefix + " " + prompt
    return prompt
