import logging
from typing import Dict, List, Tuple

import baukit as nethook
import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import ModelandTokenizer, determine_hidden_size
from src.rome import repr_tools

from .rome_hparams import ROMEHyperParams

logger = logging.getLogger(__name__)


def compute_v(
    mt: ModelandTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: List[str],
    left_vector: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    logger.info("Computing right vector (v)")

    model, tok = mt.model, mt.tokenizer

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    logger.debug(f"Lookup indices: {lookup_idxs}")

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    logger.info(f"Rewrite layer is {layer}")
    logger.info(f"Tying optimization objective to layer {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    # delta = torch.zeros(
    #     (determine_hidden_size(model),), requires_grad=True, device=mt.device
    # )
    rewrite_module = nethook.get_module(model, hparams.rewrite_module_tmp.format(layer))
    right_vector_shape = rewrite_module.weight.shape[0]
    left_vector_shape = rewrite_module.weight.shape[1]
    print(f"{right_vector_shape=} | {left_vector_shape=}")

    if hparams.mamba_block_residual:
        right_vector_shape //= 2

    delta = torch.zeros((right_vector_shape,), requires_grad=True, device=mt.device)
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.rewrite_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                logger.info("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = (
                    cur_out[0, lookup_idxs[0]][-delta.shape[0] :].detach().clone()
                )

            if hparams.mamba_block_residual:
                # this is specifically for the output of
                assert cur_layer.endswith("mixer.in_proj")
            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :][-delta.shape[0] :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    logger.debug(f"Optimizing delta of shape {delta.shape} at layer {layer}")

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.rewrite_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            output = mt(**input_tok)
            logits = output.logits if hasattr(output, "logits") else output

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        avg_prob = torch.exp(-nll_loss_each).mean().item()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )

        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        logger.info(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{avg_prob:.5f}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        if it > 12 and avg_prob > 0.95:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    # zero out the grads so that they don't accumulate and consume extra memory
    mt.model.zero_grad(set_to_none=True)

    if left_vector is None:
        logger.warning("No left vector provided. right vector ins't normalized")
        return target

    # TODO(arnab): refactor this part out of compute_v.
    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        mt=mt,
        layer=layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    if hparams.mamba_block_residual:
        n_embd_times_2 = determine_hidden_size(mt.model) * 2
        ssm_input, cur_output = cur_output.split(
            split_size=[n_embd_times_2, n_embd_times_2], dim=-1
        )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    logger.debug(f"Delta norm: {(target - cur_output).norm().item()}")
    logger.debug(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    logger.debug(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    logger.debug(f"Right vector norm: {right_vector.norm()}")

    return right_vector


def get_module_input_output_at_word(
    mt: ModelandTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        mt=mt,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        logger.debug(
            f'Lookup index found: {ret} | Sentence: {sentence} | Token:{tok.decode(tok(sentence)["input_ids"][ret])}'
        )

    return ret
