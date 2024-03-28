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
        x = x.clone()
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


from einops import einsum


# ! just ablating the diagonal ssm isn't enough.
# TODO: figure out how to ablate the shift-SSM or the conv as well
# also, the "attention" visualization is wrong, because it doesn't take the Conv into account
# technically, ssm doesn't pay attention on a particular token, it pays attention to the entire receptive field
def selective_scan_with_mask(
    self, u, delta, A, B, C, D, mask=None, retainer=None, mask_policy="subtract"
):
    """Does selective scan algorithm. See:
        - Section 2 State Space Models in the Mamba paper [1]
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]
    This is the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    Args:
        u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
        delta: shape (b, l, d_in)
        A: shape (d_in, n)
        B: shape (b, l, n)
        C: shape (b, l, n)
        D: shape (d_in,)
    Returns:
        output: shape (b, l, d_in)
    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
        Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
    """
    # **Extra Arguments**
    # ! mask: dict[target_idx] = [src_idx1, src_idx2, ...]
    # ! retainer: if an empty dictionary is passed, the retention values will be stored there
    # ! mask_policy:
    # !     - "subtract": subtracts the contribution of src from the target
    # !     - "retain": stores the retention/contribution from the source to target (retainer must be passed)

    (b, l, d_in) = u.shape

    if mask is not None:
        assert b == 1, "masking is only supported for batch size 1"
        # TODO: support batch size > 1? let's not overcomplecate things for now

    n = A.shape[1]

    # Discretize continuous parameters (A, B)
    # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
    # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
    #   "A is the more important term and the performance doesn't change much with the simplification on B"
    deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
    deltaB_u = einsum(delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n")

    # print("-" * 60)
    # print(
    #     f"{A.shape=} | {B.shape=} | {C.shape=} | {D.shape=} | {u.shape=} | {delta.shape=}"
    # )
    print(f"{deltaA.shape=} | {deltaB_u.shape=}")

    # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
    x = torch.zeros((b, d_in, n), device=deltaA.device)
    ys = []
    for i in range(l):

        # print(
        #     f"{deltaA[:, i].shape=} | {x.shape=} | {deltaB_u[:, i].shape=} | {C[:, i, :].shape=}"
        # )

        x = deltaA[:, i] * x + deltaB_u[:, i]
        y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")

        # print(f"{i} - {y.shape=}")

        ys.append(y)
    y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

    # print(y.shape)
    # print(f"{y.norm()=}")

    if mask is not None:
        for target_idx, src_idxs in mask.items():
            if retainer != None:
                retainer[target_idx] = {}
            for src_idx in src_idxs:
                assert (
                    src_idx <= target_idx
                ), f"autoregressive LM, {src_idx=} must be <= to {target_idx=}"
                delta_A_src_to_target = torch.prod(
                    deltaA[:, src_idx + 1 : target_idx + 1], dim=1
                )
                delta_B_src = deltaB_u[:, src_idx]

                # print(f"{delta_A_src_to_target.shape=}")
                # print(f"{delta_B_src.shape=}")

                delta_AB_src = delta_A_src_to_target * delta_B_src

                # print(f"{delta_AB_src.shape=}")

                retention_from_src_to_target = einsum(
                    delta_AB_src, C[:, target_idx, :], "b d_in n, b n -> b d_in"
                )

                # print(f"{retention_from_src_to_target.shape=}")

                if retainer != None:
                    retainer[target_idx][src_idx] = retention_from_src_to_target
                if mask_policy == "subtract":
                    # print(
                    #     f"subtracting {src_idx=} from {target_idx=} >> {retention_from_src_to_target.norm()}"
                    # )
                    y[:, target_idx] -= retention_from_src_to_target

    # print(y)
    print(f"{y.norm()=}")

    # # ! if the mask is everything then y at this position should be exactly zero
    if mask is not None:
        for target_idx in range(l):
            print(
                f"||y_{target_idx}|| = {y[:, target_idx].norm().item()} | IS IT ZERO: {torch.allclose(y[:, target_idx], torch.zeros_like(y[:, target_idx]), atol=1e-3)} | max = {y[:, target_idx].max().item()} | min = {y[:, target_idx].min().item()}"
            )
        print("-------------------------------------")

    y = y + u * D

    return y


# Testing code for selective_scan_with_mask

# from src.utils import experiment_utils
# experiment_utils.set_seed(123456)

# u = torch.randn(1, 4, 5120)
# delta = torch.randn(1, 4, 5120)
# A = torch.randn(5120, 16)
# B = torch.randn(1, 4, 16)
# C = torch.randn(1, 4, 16)
# D = torch.randn(5120)

# output = selective_scan_with_mask(
#     self=None,
#     u=u,
#     delta=delta,
#     A=A,
#     B=B,
#     C=C,
#     D=D,
#     # mask = {0:[]}
#     mask={0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3]},
# )

# output.shape
