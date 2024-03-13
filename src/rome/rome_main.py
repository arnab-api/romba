import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.functional import free_gpu_cache
from src.models import ModelandTokenizer
from src.rome_utils import nethook
from src.utils.generation import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .rome_hparams import ROMEHyperParams

logger = logging.getLogger(__name__)

CONTEXT_TEMPLATES_CACHE = None


# def apply_rome_to_model(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     requests: List[Dict],
#     hparams: ROMEHyperParams,
#     copy=False,
#     return_orig_weights=False,
#     cache_template: Optional[str] = None,
# ) -> Tuple[AutoModelForCausalLM, List[str]]:
#     """
#     Returns a model with the desired changes.

#     :param copy: If true, will preserve the original model while creating a new one to edit.
#         Note that you are responsible for deallocating the new model's memory to avoid leaks.

#     :return: (1) the updated model, (2) an original copy of the weights that changed
#     """

#     if copy:
#         model = deepcopy(model)

#     weights_copy = {}

#     for i, request in enumerate(requests):
#         # Caching is only valid on first request, since the model changes afterwards
#         deltas = get_kv_deltas(
#             model, tok, request, hparams, (cache_template if i == 0 else None)
#         )

#         with torch.no_grad():
#             # sequential update. each of requests can have different layers
#             # useful for checking how ROME behaves with scaling the number of requests
#             for w_name, (delta_u, delta_v) in deltas.items():
#                 upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
#                 w = nethook.get_parameter(model, w_name)
#                 upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

#                 if return_orig_weights and w_name not in weights_copy:
#                     assert i == 0
#                     weights_copy[w_name] = w.detach().clone()

#                 w[...] += upd_matrix

#         print(f"New weights successfully inserted into {list(deltas.keys())}")


#     return model, weights_copy


def restore_weights(model, weights_to_restore):
    with torch.no_grad():
        for module_name, weights in weights_to_restore.items():
            module = nethook.get_module(model, module_name)
            module.weight.copy_(weights["weight"])
            if weights["bias"] is not None:
                module.bias.copy_(weights["bias"])
    logger.info(f"restored weights of modules {list(weights_to_restore.keys())}.")


def save_original_weights(model, modules):
    module_weights = {}
    for module_name in modules:
        module = nethook.get_module(model, module_name)
        module_weights[module_name] = {
            "weight": module.weight.detach().clone(),
            "bias": module.bias.detach().clone() if module.bias is not None else None,
        }
    return module_weights


def apply_rome_to_model(
    mt: ModelandTokenizer,
    requests: Dict | List[Dict],
    hparams: ROMEHyperParams,
    cache_template: Optional[str] = None,
    return_orig_weights: bool = True,
):
    if isinstance(requests, dict):
        requests = [requests]

    if return_orig_weights:
        # save the weights for future restoration
        rewritten_modules = [
            hparams.rewrite_module_tmp.format(layer) for layer in hparams.layers
        ]
        weights_copy = save_original_weights(mt.model, rewritten_modules)

    for request in requests:
        deltas = get_kv_deltas(mt, request, hparams, cache_template)

        with torch.no_grad():
            w_name, (delta_k, delta_v) = list(deltas.items())[0]
            weights = nethook.get_parameter(mt.model, w_name)
            upd_matrix = delta_k.unsqueeze(1) @ delta_v.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights.shape)
            weights[...] += upd_matrix

    if return_orig_weights:
        return mt.model, weights_copy
    else:
        return mt.model


def get_kv_deltas(
    mt: ModelandTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        # Space required for correct tokenization
        request["target_new"]["str"] = " " + request["target_new"]["str"]
    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    # # Retrieve weights that user desires to change
    # weights = {
    #     f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
    #         mt.model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
    #     )
    #     for layer in hparams.layers
    # }
    # # Save old weights for future restoration
    # weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    hparams.layers = sorted(hparams.layers)
    for layer in hparams.layers:
        left_vector, right_vector = None, None
        require_recompute = True

        # Retrieve k/v pair if already stored in cache
        # Must be first layer, since rewrites at previous layers affect future layers
        if layer == hparams.layers[0]:
            cache_fname = (
                Path(
                    str(cache_template).format(
                        layer, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            if (
                cache_fname is not None  # Require cache template
                and cache_fname.exists()  # Cache file must exist
            ):
                try:
                    data = np.load(cache_fname)
                    left_vector = torch.from_numpy(data["left_vector"]).to("cuda")
                    right_vector = torch.from_numpy(data["right_vector"]).to("cuda")
                    require_recompute = False
                except Exception as e:
                    print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute rank-1 update matrix
        left_vector: torch.Tensor = (
            left_vector
            if left_vector is not None
            else compute_u(
                mt=mt,
                request=request,
                hparams=hparams,
                layer=layer,
                context_templates=get_context_templates(
                    mt=mt, length_params=hparams.context_template_length_params
                ),
            )
        )
        print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = (
            right_vector
            if right_vector is not None
            else compute_v(
                mt=mt,
                request=request,
                hparams=hparams,
                layer=layer,
                left_vector=left_vector,
                context_templates=get_context_templates(
                    mt, hparams.context_template_length_params
                ),
            )
        )
        logger.debug(f"Right vector shape: { right_vector.shape}")

        # Cache vectors for future use
        if cache_fname is not None and require_recompute:
            cache_fname.parent.mkdir(exist_ok=True, parents=True)
            np.savez(
                cache_fname,
                **{
                    "left_vector": left_vector.detach().cpu().numpy(),
                    "right_vector": right_vector.detach().cpu().numpy(),
                },
            )
            logger.info(f"Cached k/v pair at {cache_fname}")

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            # upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            # upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # # Update model weights and record desired changes in `delta` variable
            # weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # # Restore state of original model
    # with torch.no_grad():
    #     for k, v in weights.items():
    #         v[...] = weights_copy[k]

    logger.info(f"Deltas successfully computed for {weight_name}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(
    mt: ModelandTokenizer, length_params: list[tuple]
) -> list[str]:
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x + ". {}"
            for x in sum(
                (
                    generate_fast(
                        mt=mt,
                        prompts=["<|endoftext|>"],
                        n_gen_per_prompt=n_gen,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
