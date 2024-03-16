import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, overload

import baukit
import torch
import transformers

# from mamba_ssm.ops.triton.layernorm import rms_norm_fn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.typing import Mamba

logger = logging.getLogger(__name__)


CACHEABLE_FUNCS = ["forward", "ssm", "selective_scan"]


class ModelandTokenizer:
    def __init__(
        self,
        model: Optional[transformers.AutoModel] = None,
        tokenizer: Optional[transformers.AutoTokenizer] = None,
        model_path: Optional[
            str
        ] = "EleutherAI/gpt-j-6B",  # if model is provided, this will be ignored and rewritten
        torch_dtype=torch.float32,
    ) -> None:
        assert (
            model is not None or model_path is not None
        ), "Either model or model_name must be provided"
        if model is not None:
            assert tokenizer is not None, "Tokenizer must be provided with the model"
            self.name = model.config._name_or_path
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if "mamba" in model_path.lower():
                model = Mamba.from_pretrained(model_path).to(torch_dtype).to("cuda")
                tokenizer = AutoTokenizer.from_pretrained(
                    "EleutherAI/gpt-neox-20b",  # Mamba was trained on the Pile with this exact tokenizer
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    # torch_dtype=torch_dtype,
                ).to(device)

                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    # padding_side='left'
                )

            tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            logger.info(
                f"loaded model <{model_path}> | size: {get_model_size(model)} | dtype: {torch_dtype} | device: {device}"
            )
            self.name = model_path

        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.device = next(self.model.parameters()).device

        (
            self.parse_config()
            if isinstance(model, Mamba)
            else self.parse_config(model.config)
        )
        self.cache_forwards()

    def parse_config(self, model_config=None) -> None:
        fields = {
            "n_layer": None,
            "n_embd": None,
            "layer_name_format": None,
            "layer_names": None,
            "embedder_name": None,
            "final_layer_norm_name": None,
            "lm_head_name": None,
        }

        fields["n_layer"] = len(determine_layers(self))
        fields["n_embd"] = determine_hidden_size(self)
        fields["embedder_name"] = determine_embedding_layer_path(self)
        fields["final_layer_norm_name"] = determine_final_layer_norm_path(self)
        fields["lm_head_name"] = determine_lm_head_path(self)
        fields["layer_name_format"] = determine_layer_name_format(self)

        if is_mamba_variant(self) == False:
            fields["attn_module_name_format"] = None
            fields["mlp_module_name_format"] = None
            if is_llama_variant(self.model):
                fields["mlp_module_name_format"] = "model.layers.{}.mlp"
                fields["attn_module_name_format"] = "model.layers.{}.self_attn"

            elif is_gpt_variant(self.model):
                # ! will be a little different for neox models. Ignoring for now
                fields["mlp_module_name_format"] = "transformer.h.{}.mlp"
                fields["attn_module_name_format"] = "transformer.h.{}.attn"

            elif is_pythia_variant(self.model):
                fields["mlp_module_name_format"] = "gpt_neox.layers.{}.mlp"
                fields["attn_module_name_format"] = "gpt_neox.layers.{}.attention"

            else:
                logger.error(f"Unknown model type: {type(self.model).__name__}")

        if fields["layer_name_format"] is not None and fields["n_layer"] is not None:
            fields["layer_names"] = [
                fields["layer_name_format"].format(i) for i in range(fields["n_layer"])
            ]

        for key, value in fields.items():
            if value is None:
                logger.error(
                    f"!!! Error ({type(self.model).__name__}): {key} could not be set !!!"
                )
            setattr(self, key, value)

    @property
    def lm_head(self) -> torch.nn.Sequential:
        lm_head = baukit.get_module(self.model, self.lm_head_name)
        ln_f = baukit.get_module(self.model, self.final_layer_norm_name)
        # ln_f = FinalLayerNorm(ln_f, mamba=isinstance(self.model, Mamba))
        return LMHead(final_layer_norm=ln_f, lm_head=lm_head)

    def cache_forwards(self):
        """
        Caches the forward pass of all the modules.
        Usuful to reset the model to its original state after an overwrite.
        """
        self._module_forwards: dict = {}
        for name, module in self.model.named_modules():
            self._module_forwards[name] = {}
            for func_name in CACHEABLE_FUNCS:
                if hasattr(module, func_name):
                    self._module_forwards[name][func_name] = getattr(module, func_name)

    def reset_forward(self) -> None:
        """
        Resets the forward pass of all the modules to their original state.
        """
        for name, module in self.model.named_modules():
            for func_name in CACHEABLE_FUNCS:
                if hasattr(module, func_name):
                    setattr(module, func_name, self._module_forwards[name][func_name])

    def __call__(self, *args, **kwargs) -> Any:
        """Call the model."""
        # print(f"{self.is_mamba=} | {self.is_mamba_fast=}")
        if is_mamba_variant(self):  # Mamba can only handle input_ids
            for k in list(kwargs.keys()):
                if k.startswith("input") == False:
                    kwargs.pop(k)
        return self.model(*args, **kwargs)


class LMHead(torch.nn.Module):
    def __init__(self, final_layer_norm: torch.nn.Module, lm_head: torch.nn.Module):
        super().__init__()
        self.lm_head = lm_head
        self.final_layer_norm = final_layer_norm

    def forward(
        self,
        x: torch.Tensor,
        # residual: Optional[torch.Tensor] = None
    ):
        x = self.final_layer_norm(
            x,
            # residual
        )
        return self.lm_head(x)


def get_model_size(
    model: torch.nn.Module, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all = param_size + buffer_size

    return bytes_to_human_readable(size_all, unit)


def bytes_to_human_readable(
    size: int, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> str:
    denom = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[unit]
    return f"{size / denom:.3f} {unit}"


def unwrap_model(mt: ModelandTokenizer | torch.nn.Module) -> torch.nn.Module:
    if isinstance(mt, ModelandTokenizer):
        return mt.model
    if isinstance(mt, torch.nn.Module):
        return mt
    raise ValueError("mt must be a ModelandTokenizer or a torch.nn.Module")


def unwrap_tokenizer(mt: ModelandTokenizer | AutoTokenizer) -> AutoTokenizer:
    if isinstance(mt, ModelandTokenizer):
        return mt.tokenizer
    return mt


def untuple(object: Any):
    if isinstance(object, tuple):
        return object[0]
    return object


from src.utils.typing import Model, Tokenizer


def maybe_prefix_eos(tokenizer, prompt: str) -> str:
    """Prefix prompt with EOS token if model has no special start token."""
    tokenizer = unwrap_tokenizer(tokenizer)
    if hasattr(tokenizer, "eos_token"):
        prefix = tokenizer.eos_token
        if not prompt.startswith(prefix):
            prompt = prefix + " " + prompt
    return prompt


def is_pythia_variant(model: Model | ModelandTokenizer) -> bool:
    """Determine if model is GPT variant."""
    if isinstance(model, ModelandTokenizer):
        model = unwrap_model(model)
    try:
        return "pythia" in model.config._name_or_path.lower()
    except:
        return False


def is_gpt_variant(mt: Model | ModelandTokenizer) -> bool:
    """Determine if model/tokenizer is GPT variant."""
    if isinstance(mt, ModelandTokenizer):
        mt = unwrap_model(mt)

    # pythia models also have GPTNeoXForCausalLM architecture, but they have slightly  different structure
    # so we need to check for them separately
    if is_pythia_variant(mt):
        return False
    return isinstance(
        mt,
        transformers.GPT2LMHeadModel
        | transformers.GPTJForCausalLM
        | transformers.GPTNeoForCausalLM
        | transformers.GPTNeoXForCausalLM
        | transformers.GPT2TokenizerFast
        | transformers.GPTNeoXTokenizerFast,
    )


def is_llama_variant(mt: Model | ModelandTokenizer) -> bool:
    """Determine if model/tokenizer is GPT variant."""
    if isinstance(mt, ModelandTokenizer):
        mt = unwrap_model(mt)
    if isinstance(mt, transformers.LlamaForCausalLM):
        return True
    if hasattr(mt, "config"):
        config = mt.config
        if hasattr(config, "_name_or_path"):
            name = config._name_or_path
            return "llama" in name.lower() or "mistral" in name.lower()
    return False


def is_mamba_variant(mt: Model | ModelandTokenizer) -> bool:
    """Determine if model/tokenizer is GPT variant."""
    if isinstance(mt, ModelandTokenizer):
        if "mamba" in mt.name.lower():
            return True
        mt = unwrap_model(mt)
    return isinstance(mt, Mamba)


def is_mamba_fast(mt: ModelandTokenizer) -> bool:
    """Determine if model/tokenizer is GPT variant."""
    if isinstance(mt, ModelandTokenizer):
        mt = unwrap_model(mt)
    return is_mamba_variant(mt) and hasattr(mt, "backbone")


def any_parameter(model: ModelandTokenizer | Model) -> torch.nn.Parameter | None:
    """Get any example parameter for the model."""
    model = unwrap_model(model)
    return next(iter(model.parameters()), None)


def determine_embedding_layer_path(model: ModelandTokenizer | Model) -> str:
    model = unwrap_model(model)
    if is_gpt_variant(model):
        return "transformer.wte"
    elif isinstance(model, transformers.LlamaForCausalLM):
        return "model.embed_tokens"
    elif isinstance(model, Mamba):
        prefix = "backbone." if hasattr(model, "backbone") else ""
        return prefix + "embedding"
    elif is_pythia_variant(model):
        return "gpt_neox.embed_in"
    else:
        raise ValueError(f"unknown model type: {type(model).__name__}")


def determine_final_layer_norm_path(model: ModelandTokenizer | Model) -> str:
    model = unwrap_model(model)
    if is_gpt_variant(model):
        return "transformer.ln_f"
    elif isinstance(model, transformers.LlamaForCausalLM):
        return "model.norm"
    elif isinstance(model, Mamba):
        prefix = "backbone." if hasattr(model, "backbone") else ""
        return prefix + "norm_f"
    elif is_pythia_variant(model):
        return "gpt_neox.final_layer_norm"
    else:
        raise ValueError(f"unknown model type: {type(model).__name__}")


def determine_lm_head_path(model: ModelandTokenizer | Model) -> str:
    model = unwrap_model(model)
    if is_gpt_variant(model):
        return "lm_head"
    elif isinstance(model, transformers.LlamaForCausalLM):
        return "model.lm_head"
    elif isinstance(model, Mamba):
        return "lm_head"
    elif is_pythia_variant(model):
        return "embed_out"
    else:
        raise ValueError(f"unknown model type: {type(model).__name__}")


def determine_layers(model: ModelandTokenizer | Model) -> tuple[int, ...]:
    """Return all hidden layer names for the given model."""
    model = unwrap_model(model)
    assert isinstance(model, Model)

    if isinstance(
        model, transformers.GPTNeoXForCausalLM | transformers.LlamaForCausalLM
    ):
        n_layer = model.config.num_hidden_layers
    elif isinstance(model, Mamba):
        n_layer = (
            len(model.backbone.layers)
            if hasattr(model, "backbone")
            else len(model.layers)
        )
    else:
        n_layer = model.config.n_layer

    return (*range(n_layer),)


from src.utils.typing import Layer, Sequence


def determine_layer_name_format(
    model: ModelandTokenizer | Model,
) -> str | None:
    """Determine the format of layer names."""
    model = unwrap_model(model)

    if is_gpt_variant(model):
        if isinstance(model, transformers.GPTNeoXForCausalLM):
            return "gpt_neox.layers.{}"
        return "transformer.h.{}"
    elif is_llama_variant(model):
        return "model.layers.{}"
    elif is_pythia_variant(model):
        return "gpt_neox.layers.{}"
    elif is_mamba_variant(model):
        prefix = "backbone." if hasattr(model, "backbone") else ""
        return prefix + "layers.{}"


@overload
def determine_layer_paths(
    model: ModelandTokenizer | Model,
    layers: Optional[Sequence[Layer]] = ...,
    *,
    return_dict: Literal[False] = ...,
) -> Sequence[str]:
    """Determine layer path for each layer."""
    ...


@overload
def determine_layer_paths(
    model: ModelandTokenizer | Model,
    layers: Optional[Sequence[Layer]] = ...,
    *,
    return_dict: Literal[True],
) -> dict[Layer, str]:
    """Determine mapping from layer to layer path."""
    ...


def determine_layer_paths(
    model: ModelandTokenizer | Model,
    layers: Optional[Sequence[Layer]] = None,
    *,
    return_dict: bool = False,
) -> Sequence[str] | dict[Layer, str]:
    """Determine the absolute paths to the given layers in the model.

    Args:
        model: The model.
        layers: The specific layer (numbers/"emb") to look at. Defaults to all of them.
            Can be a negative number.
        return_dict: If True, return mapping from layer to layer path,
            otherwise just return list of layer paths in same order as `layers`.

    Returns:
        Mapping from layer number to layer path.

    """
    model = unwrap_model(model)

    if layers is None:
        layers = determine_layers(model)

    assert isinstance(model, Model), type(model)

    layer_paths: dict[Layer, str] = {}
    layer_name_format = determine_layer_name_format(model)
    for layer in layers:
        if layer == "emb":
            layer_paths[layer] = determine_embedding_layer_path(model)
            continue
        if layer == "ln_f":
            layer_paths[layer] = determine_final_layer_norm_path(model)
            continue

        layer_index = layer
        if layer_index < 0:
            layer_index = len(determine_layers(model)) + layer

        layer_paths[layer] = layer_name_format.format(layer_index)

    return layer_paths if return_dict else tuple(layer_paths[la] for la in layers)


def determine_hidden_size(model: ModelandTokenizer | Model) -> int:
    """Determine hidden rep size for the model."""
    model = unwrap_model(model)

    if isinstance(model, Mamba):
        embed = baukit.get_module(model, determine_embedding_layer_path(model))
        return embed.weight.shape[-1]

    return model.config.hidden_size


def determine_device(model: ModelandTokenizer | Model) -> torch.device | None:
    """Determine device model is running on."""
    parameter = any_parameter(model)
    return parameter.device if parameter is not None else None


def determine_dtype(model: ModelandTokenizer | Model) -> torch.dtype | None:
    """Determine dtype of model."""
    parameter = any_parameter(model)
    return parameter.dtype if parameter is not None else None
