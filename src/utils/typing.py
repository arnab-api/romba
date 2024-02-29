"""Some useful type aliases relevant to this project."""

import pathlib
from typing import Literal, Sequence

import numpy
import torch
import transformers
import transformers.modeling_outputs

from mamba_minimal.model import (
    Mamba,
)  # `mamba-simple`, easier to understand the do different interventions

# from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel as Mamba # official implementation, superfast


ArrayLike = list | tuple | numpy.ndarray | torch.Tensor
PathLike = str | pathlib.Path
Device = str | torch.device

# Throughout this codebase, we use HuggingFace model implementations.
Model = (
    transformers.GPT2LMHeadModel
    | transformers.GPTJForCausalLM
    | transformers.GPTNeoXForCausalLM
    | transformers.LlamaForCausalLM
    | Mamba
)
Tokenizer = transformers.PreTrainedTokenizerFast
TokenizerOffsetMapping = Sequence[tuple[int, int]]
ModelInput = transformers.BatchEncoding
ModelOutput = transformers.modeling_outputs.CausalLMOutput
ModelGenerateOutput = transformers.generation.utils.GenerateOutput | torch.LongTensor

Layer = int | Literal["emb"] | Literal["ln_f"]

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings.
StrSequence = list[str] | tuple[str, ...]
