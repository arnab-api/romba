import logging
from typing import Callable, Literal, Optional, get_args

import torch
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


MambaBlock_Hook_Points = Literal[
    "ssm_after_up_proj",
    "ssm_after_conv1D",
    "ssm_after_silu",
    "ssm_after_ssm",
    "mlp_after_up_proj",
    "mlp_after_silu",
    "before_down_proj",
    "after_down_proj",  # the output of the mamba block #! Not the residual
]


def MambaBlockForwardPatcher(
    patch_spec: Optional[dict[int, torch.Tensor]] = None,
    patch_hook: Optional[
        MambaBlock_Hook_Points
    ] = None,  # If None => do not patch, return the original output
    retainer: Optional[
        dict
    ] = None,  # if a dictionary is passed, will retain all the activations[patch_idx] at different hook points
) -> Callable:
    # TODO: Assumes a single prompt for now. Should we consider batching?
    """
    Returns a replacement for the `forward()` method of `MambaBlock` to patch activations at different steps.
    """
    if patch_hook is None:
        assert (
            patch_spec is None
        ), "Need to specify `patch_hook` if `patch_spec` is not None"
    else:
        assert patch_hook in get_args(
            MambaBlock_Hook_Points
        ), f"Unknown `{patch_hook=}`, should be one of {get_args(MambaBlock_Hook_Points)}"
        assert isinstance(
            patch_spec, dict
        ), f"Need to specify `patch_spec` as a dictionary for `{patch_hook=}`"
    if retainer is not None:
        assert isinstance(retainer, dict)

    def forward_patcher(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(
            split_size=[self.args.d_inner, self.args.d_inner], dim=-1
        )
        # ------------------------------------------------------
        if patch_hook == "ssm_after_up_proj":
            for patch_idx, patch_vector in patch_spec.items():
                x[:, patch_idx] = patch_vector.to(x.dtype).to(x.device)
        elif patch_hook == "mlp_after_up_proj":
            for patch_idx, patch_vector in patch_spec.items():
                res[:, patch_idx] = patch_vector.to(res.dtype).to(res.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["ssm_after_up_proj"] = x.detach().clone()
            retainer["mlp_after_up_proj"] = res.detach().clone()
        # ------------------------------------------------------

        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d_in l -> b l d_in")
        # ------------------------------------------------------
        if patch_hook == "ssm_after_conv1D":
            for patch_idx, patch_vector in patch_spec.items():
                x[:, patch_idx] = patch_vector.to(x.dtype).to(x.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["ssm_after_conv1D"] = x.detach().clone()
        # ------------------------------------------------------

        x = F.silu(x)
        # ------------------------------------------------------
        if patch_hook == "ssm_after_silu":
            for patch_idx, patch_vector in patch_spec.items():
                x[:, patch_idx] = patch_vector.to(x.dtype).to(x.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["ssm_after_silu"] = x.detach().clone()
        # ------------------------------------------------------

        y = self.ssm(x)
        # ------------------------------------------------------
        if patch_hook == "ssm_after_ssm":
            for patch_idx, patch_vector in patch_spec.items():
                y[:, patch_idx] = patch_vector.to(y.dtype).to(y.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["ssm_after_ssm"] = y.detach().clone()
        # ------------------------------------------------------

        res = F.silu(res)
        # ------------------------------------------------------
        if patch_hook == "mlp_after_silu":
            for patch_idx, patch_vector in patch_spec.items():
                res[:, patch_idx] = patch_vector.to(res.dtype).to(res.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["mlp_after_silu"] = res.detach().clone()
        # ------------------------------------------------------

        y = y * res
        # ------------------------------------------------------
        if patch_hook == "before_down_proj":
            for patch_idx, patch_vector in patch_spec.items():
                y[:, patch_idx] = patch_vector.to(y.dtype).to(y.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["before_down_proj"] = y.detach().clone()
        # ------------------------------------------------------

        output = self.out_proj(y)
        # ------------------------------------------------------
        if patch_hook == "after_down_proj":
            for patch_idx, patch_vector in patch_spec.items():
                output[:, patch_idx] = patch_vector.to(output.dtype).to(output.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["after_down_proj"] = output.detach().clone()
        # ------------------------------------------------------

        return output

    return forward_patcher
