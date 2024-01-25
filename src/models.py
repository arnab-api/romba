import logging
import re
from typing import Any, Callable, Literal, Optional

import baukit
import torch
import transformers

# from mamba_ssm.ops.triton.layernorm import rms_norm_fn
from transformers import AutoModelForCausalLM, AutoTokenizer

# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel as Mamba
# use `mamba-simple`, the official implementation is to messy
from mamba_minimal.model import Mamba

logger = logging.getLogger(__name__)


class ModelandTokenizer:
    def __init__(
        self,
        model: Optional[transformers.AutoModel] = None,
        tokenizer: Optional[transformers.AutoTokenizer] = None,
        model_path: Optional[
            str
        ] = "EleutherAI/gpt-j-6B",  # if model is provided, this will be ignored and rewritten
        torch_dtype=torch.float16,
    ) -> None:
        assert (
            model is not None or model_path is not None
        ), "Either model or model_name must be provided"
        if model is not None:
            assert tokenizer is not None, "Tokenizer must be provided with the model"
            self.model_name = model.config._name_or_path
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if "mamba" in model_path.lower():
                model = Mamba.from_pretrained(model_path).to(torch_dtype).to("cuda")
                tokenizer = AutoTokenizer.from_pretrained(
                    "EleutherAI/gpt-neox-20b",  # Mamba was trained on the Pile with this exact tokenizer
                )
            else:
                model, tokenizer = (
                    AutoModelForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch_dtype,
                    ).to(device),
                    AutoTokenizer.from_pretrained(
                        model_path,
                        # padding_side='left'
                    ),
                )

            tokenizer.pad_token = tokenizer.eos_token
            model.eval()

            logger.info(
                f"loaded model <{model_path}> | size: {get_model_size(model) :.3f} MB | dtype: {torch_dtype} | device: {device}"
            )
            self.model_name = model_path

        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.parse_config() if isinstance(model, Mamba) else self.parse_config(
            model.config
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
        if isinstance(self.model, Mamba):  # Not a Transformer
            fields["n_layer"] = len(self.model.layers)
            fields["n_embd"] = self.model.embedding.weight.shape[-1]
            fields["layer_name_format"] = "layers.{}"
            fields["embedder_name"] = "embedding"
            fields["final_layer_norm_name"] = "norm_f"
            fields["lm_head_name"] = "lm_head"
        else:
            fields["attn_module_name_format"] = None
            fields["mlp_module_name_format"] = None
            if (
                "mistral" in model_config._name_or_path.lower()
                or "llama" in model_config._name_or_path.lower()
            ):
                fields["n_layer"] = model_config.num_hidden_layers
                fields["n_embd"] = model_config.hidden_size
                fields["layer_name_format"] = "model.layers.{}"
                fields["mlp_module_name_format"] = "model.layers.{}.mlp"
                fields["attn_module_name_format"] = "model.layers.{}.self_attn"
                fields["embedder_name"] = "model.embed_tokens"
                fields["final_layer_norm_name"] = "model.norm"
                fields["lm_head_name"] = "model.lm_head"

            elif "gpt-j" in model_config._name_or_path.lower():
                fields["n_layer"] = model_config.n_layer
                fields["n_embd"] = model_config.n_embd
                fields["layer_name_format"] = "transformer.h.{}"
                fields["mlp_module_name_format"] = "transformer.h.{}.mlp"
                fields["attn_module_name_format"] = "transformer.h.{}.attn"
                fields["embedder_name"] = "transformer.wte"
                fields["final_layer_norm_name"] = "transformer.ln_f"
                fields["lm_head_name"] = "transformer.lm_head"

        if fields["layer_name_format"] is not None and fields["n_layer"] is not None:
            fields["layer_names"] = [
                fields["layer_name_format"].format(i) for i in range(fields["n_layer"])
            ]

        for key, value in fields.items():
            if value is None:
                print(f"!!! Warning: {key} could not be set !!!")
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
        self._module_forwards: dict(
            str,
        ) = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "forward"):
                self._module_forwards[name] = module.forward

    def reset_forward(self) -> None:
        """
        Resets the forward pass of all the modules to their original state.
        """
        for name, module in self.model.named_modules():
            if hasattr(module, "forward"):
                module.forward = self._module_forwards[name]


# class FinalLayerNorm(torch.nn.Module):
#     def __init__(self, ln_f: torch.nn.Module, mamba: bool = False):
#         super().__init__()
#         self.ln_f = ln_f
#         self.mamba = mamba

#     def forward(self, x: torch.Tensor, residual=Optional[torch.Tensor]):
#         if self.mamba == False:
#             return self.ln_f(untuple(x))
#         else:
#             if residual is None:
#                 try:
#                     x, residual = x
#                 except:
#                     raise ValueError("x must be a tuple of (x, residual)")
#             return rms_norm_fn(
#                 x=x,
#                 weight=self.ln_f.weight,
#                 bias=self.ln_f.bias,
#                 eps=self.ln_f.eps,
#                 residual=residual,
#                 prenorm=False,
#                 residual_in_fp32=self.ln_f.weight.dtype == torch.float32,
#             )


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
    denom = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[unit]
    return size_all / denom


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
