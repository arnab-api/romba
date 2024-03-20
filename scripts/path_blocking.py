import argparse
import copy
import json
import logging
import os
import types
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Literal, Optional, get_args

import baukit
import numpy
import torch
from dataclasses_json import DataClassJsonMixin
from tqdm import tqdm

import src.tokens as tokenizer_utils

# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel as Mamba
from mamba_minimal.model import Mamba
from src import functional
from src.dataset.dataclasses import load_relation
from src.dataset.rome_dataclasses import CounterFactDataset
from src.functional import (
    decode_tokens,
    filter_samples_by_model_knowledge,
    find_token_range,
    make_inputs,
    predict_from_input,
)
from src.globals import DATA_DIR, RESULTS_DIR
from src.hooking.mamba import MambaBlock_Hook_Points, MambaBlockForwardPatcher
from src.models import ModelandTokenizer, is_mamba_variant
from src.tracing import calculate_average_indirect_effects
from src.utils import logging_utils

logger = logging.getLogger(__name__)


def path_ablation(
    mt,
    inp,  # A set of inputs
    residual_states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    alt_subj_patching: bool = False,  # If True, will assume inp shape to be (2, L). Uncorrupted activations with inp[0] will be patched in the run with inp[1]. Will not corrupt the embeddings
    block_states_to_unpatch: (
        list
    ) = [],  # A list of (token index, layername) triples to restore in the uncorrupted run
    hook_to_unpatch: Optional[MambaBlock_Hook_Points] = None,
):
    assert is_mamba_variant(
        mt
    ), "This function is only for Mamba models. check trace_with_repatch in the original implementation for other models"
    embed_layername = mt.embedder_name

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    residual_patch_spec = defaultdict(list)
    for t, l in residual_states_to_patch:
        residual_patch_spec[l].append(t)

    block_unpatch_spec = defaultdict(list)
    for t, l in block_states_to_unpatch:
        block_unpatch_spec[l].append(t)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(repr, layer):
        assert first_run_residual_activations is not None

        if layer == embed_layername and alt_subj_patching == False:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(repr.shape[0] - 1, e - b, repr.shape[2]))
                ).to(repr.device)
                if replace:
                    repr[1:, b:e] = noise_data
                else:
                    repr[1:, b:e] += noise_data
            return repr

        if layer not in residual_patch_spec:
            return repr

        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens from the first run
        h = untuple(repr)
        for t in residual_patch_spec[layer]:
            h[1:, t] = untuple(first_run_residual_activations[layer].output)[0, t]

        return repr

    mt.reset_forward()  # reset the model to use default forward functions

    # need to run twice to store corrupted and uncorrupted activations
    # of hooks inside the MambaBlock (Is there a better way?)
    interested_layers = list(
        set(list(residual_patch_spec.keys()) + list(block_unpatch_spec.keys()))
    )
    # print(f"{interested_layers=}")
    first_run_hook_activations = {layer: {} for layer in interested_layers}
    # print(f"{first_run_hook_activations.keys()=}")
    for layer in interested_layers:
        block = baukit.get_module(
            mt.model, name=layer + ".mixer"
        )  # MambaBlock naming format
        block.forward = types.MethodType(
            MambaBlockForwardPatcher(
                retainer=first_run_hook_activations[layer],
            ),  # get everything for the uncorrupted run
            block,
        )
    with torch.inference_mode(), baukit.TraceDict(
        mt.model,
        [embed_layername] + list(residual_patch_spec.keys()),
        edit_output=None,  # No intervention on the clean run
    ) as td:
        mt(**inp)

    first_run_residual_activations = td

    # print(f"{first_run_residual_activations.keys()=}")

    # ------------------------------------------------------
    # second run with patching / unpatching
    mt.reset_forward()  # reset the model to use default forward functions
    for layer in block_unpatch_spec:
        block = baukit.get_module(mt.model, name=layer + ".mixer")

        # restore the corrupted activations inside the MambaBlock here
        # the patching with clean residual states will be done in patch_rep
        cur_patch_spec = {
            token_idx: first_run_hook_activations[layer][hook_to_unpatch][
                1, t
            ]  # corrupted activations
            for token_idx in block_unpatch_spec[layer]
        }
        block.forward = types.MethodType(
            MambaBlockForwardPatcher(
                patch_spec=cur_patch_spec,
                patch_hook=hook_to_unpatch,
            ),
            block,
        )
    with torch.inference_mode(), baukit.TraceDict(
        mt.model,
        [embed_layername]
        + list(
            residual_patch_spec.keys()
        ),  # make sure to patch from the clean activations
        edit_output=patch_rep,  # passing to patch_rep to noise the embeddings only. Restoring the states is done in the MambaBlockForwardPatcher forwards
    ):
        outputs_exp = mt.model(input_ids=inp["input_ids"])
    # ------------------------------------------------------
    mt.reset_forward()  # reset the model to use default forward functions

    # We report softmax probabilities for the answers_t token predictions of interest.
    logits = outputs_exp.logits if hasattr(outputs_exp, "logits") else outputs_exp
    probs = torch.softmax(logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs


def trace_important_states_with_ablation(
    mt,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
    ablate_mambahook: Optional[MambaBlock_Hook_Points] = None,
    alt_subj_patching: bool = False,
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    if token_range is None:
        token_range = range(ntoks)

    for tnum in token_range:
        block_states_to_unpatch = []
        if ablate_mambahook is not None:
            block_states_to_unpatch = [
                (tnum, mt.layer_name_format.format(layer))
                for layer in range(0, mt.n_layer)
            ]

        row = []
        for layer in range(mt.n_layer):
            r = path_ablation(
                mt=mt,
                inp=inp,
                residual_states_to_patch=[(tnum, mt.layer_name_format.format(layer))],
                answers_t=answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
                block_states_to_unpatch=block_states_to_unpatch,
                hook_to_unpatch=ablate_mambahook,
                alt_subj_patching=alt_subj_patching,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


from src.tracing import replace_eos_with_pad, trace_with_patch


def calculate_hidden_flow_with_ablation(
    mt: ModelandTokenizer,
    prompt: str,
    subject: str,
    alt_subject: Optional[str] = None,
    num_samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    ablate_mambahook: Optional[MambaBlock_Hook_Points] = None,
):
    if alt_subject is None:
        inp = make_inputs(mt.tokenizer, [prompt] * (num_samples + 1))
        with torch.no_grad():
            answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
        e_range = find_token_range(
            string=prompt,
            substring=subject,
            tokenizer=mt.tokenizer,
        )
        low_score = trace_with_patch(
            mt,
            inp,
            [],
            answer_t,
            e_range,
            noise=noise,
            uniform_noise=uniform_noise,
            mamba_block_hook=None,  # don't need to patch for calculating the low score
            alt_subj_patching=alt_subject is not None,
        ).item()
    else:
        if "{}" in prompt:
            prompt = prompt.format(subject)
        clean_prompt = prompt
        alt_prompt = prompt.replace(subject, alt_subject)
        with tokenizer_utils.set_padding_side(mt.tokenizer, padding_side="left"):
            inp = mt.tokenizer(
                [clean_prompt, alt_prompt],
                return_tensors="pt",
                padding="longest",
                return_offsets_mapping=True,
            ).to(mt.device)
        offset_mapping = inp.pop("offset_mapping")
        subject_range = find_token_range(
            string=clean_prompt,
            substring=subject,
            tokenizer=mt.tokenizer,
            offset_mapping=offset_mapping[0],
        )
        alt_subj_range = find_token_range(
            string=alt_prompt,
            substring=alt_subject,
            tokenizer=mt.tokenizer,
            offset_mapping=offset_mapping[1],
        )
        assert subject_range[1] == alt_subj_range[1]
        e_range = (min(subject_range[0], alt_subj_range[0]), subject_range[1])

        with torch.no_grad():
            outputs = mt(**inp)
        logits = outputs.logits[:, -1] if hasattr(outputs, "logits") else outputs[:, -1]
        next_token_probs = logits.float().softmax(dim=-1)
        answer_t = next_token_probs[0].argmax(dim=-1)
        base_score = next_token_probs[0, answer_t]  # p(ans|subj)
        low_score = next_token_probs[1, answer_t]  # p(ans|alt_subj)

    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range == "prompt_last":
        token_range = [inp["input_ids"].shape[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")

    [answer] = decode_tokens(mt.tokenizer, [answer_t])

    differences = trace_important_states_with_ablation(
        mt=mt,
        inp=inp,
        e_range=e_range,
        answer_t=answer_t,
        noise=noise,
        uniform_noise=uniform_noise,
        replace=replace,
        token_range=token_range,
        ablate_mambahook=ablate_mambahook,
        alt_subj_patching=alt_subject is not None,
    )

    differences = differences.detach().cpu()
    indirect_effect = dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=replace_eos_with_pad(
            mt.tokenizer, list(decode_tokens(mt.tokenizer, inp["input_ids"][0]))
        ),
        subject_range=e_range,
        answer=answer,
        window=window,
        correct_prediction=True,
        kind=ablate_mambahook,
    )

    if alt_subject is not None:
        indirect_effect["alt_subject"] = replace_eos_with_pad(
            mt.tokenizer,
            list(
                decode_tokens(
                    mt.tokenizer, inp["input_ids"][1, e_range[0] : e_range[1]]
                )
            ),
        )

    return indirect_effect


from src.dataset.dataclasses import RelationSample


@dataclass
class AblationResult(DataClassJsonMixin):
    sample: RelationSample
    alt_sample: RelationSample
    prompt_template: str
    patch_recovery: list[float]
    ssm_severed: list[float]
    mlp_severed: list[float]


@dataclass
class AblationResultsAll(DataClassJsonMixin):
    relation_name: str
    at_token: str
    trials: list[AblationResult] = field(default_factory=list)


def run_ablation_experiment(
    model_path: str,
    trials_per_relation: Optional[int] = None,
    relation_names: list[str] = [
        "place_in_city",
        "country_capital_city",
        "person_occupation",
        "person_plays_pro_sport",
        "company_ceo",
        "company_hq",
        "person_native_language",
        "landmark_in_country",
    ],
):
    mt = ModelandTokenizer(model_path=model_path)
    relation_dir = os.path.join(DATA_DIR, "relation", "factual")
    results_dir = os.path.join(RESULTS_DIR, "ablation")
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to {results_dir}")
    for relation_name in relation_names:
        logger.info("-" * 50)
        logger.info(f"Processing relation => {relation_name}")
        relation = load_relation(
            file=os.path.join(relation_dir, f"{relation_name}.json")
        )
        relation.select_icl_examples(0)
        relation = filter_samples_by_model_knowledge(
            mt=mt,
            relation=relation,
            # limit=trials_per_relation,
        )

        edit_targets = functional.random_edit_targets(samples=relation.samples)
        samples = (
            relation.samples[: min(trials_per_relation, len(relation.samples))]
            if trials_per_relation is not None
            else relation.samples
        )
        prompt_template = relation.prompt_templates[0]

        relation_path = os.path.join(results_dir, f"{relation_name}")
        os.makedirs(relation_path, exist_ok=True)

        for setting in ["subject_last", "prompt_last"]:
            common_kwargs = dict(
                mt=mt,
                prompt=prompt_template,
                token_range=setting,
            )

            ablation_results = AblationResultsAll(
                relation_name=relation_name,
                at_token=setting,
                trials=[],
            )

            for sample in tqdm(samples):
                alt_sample = edit_targets[sample]
                print(f"sample={str(sample)}, alt_sample={str(alt_sample)}")

                indirect_effect = calculate_hidden_flow_with_ablation(
                    subject=sample.subject,
                    alt_subject=alt_sample.subject,
                    ablate_mambahook=None,
                    **common_kwargs,
                )

                indirect_effect_ssm_severed = calculate_hidden_flow_with_ablation(
                    subject=sample.subject,
                    alt_subject=alt_sample.subject,
                    ablate_mambahook="ssm_after_ssm",
                    **common_kwargs,
                )

                indirect_effect_mlp_severed = calculate_hidden_flow_with_ablation(
                    subject=sample.subject,
                    alt_subject=alt_sample.subject,
                    ablate_mambahook="mlp_after_silu",
                    **common_kwargs,
                )

                high_score = indirect_effect["high_score"].item()
                low_score = indirect_effect["low_score"].item()

                patch_recovery = (indirect_effect["scores"] - low_score) / (
                    high_score - low_score
                )
                ssm_severed_recovery = (
                    indirect_effect_ssm_severed["scores"] - low_score
                ) / (high_score - low_score)
                mlp_severed_recovery = (
                    indirect_effect_mlp_severed["scores"] - low_score
                ) / (high_score - low_score)

                ablation_results.trials.append(
                    AblationResult(
                        sample=sample,
                        alt_sample=alt_sample,
                        prompt_template=prompt_template,
                        patch_recovery=patch_recovery.tolist(),
                        ssm_severed=ssm_severed_recovery.tolist(),
                        mlp_severed=mlp_severed_recovery.tolist(),
                    )
                )

                with open(os.path.join(relation_path, f"{setting}.json"), "w") as f:
                    json.dump(ablation_results.to_dict(), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging_utils.add_logging_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "state-spaces/mamba-2.8b",
            "EleutherAI/pythia-2.8b-deduped",
        ],
    )
    parser.add_argument("--n_trial", type=int, default=None)
    parser.add_argument("--relation", type=str, default=None)

    args = parser.parse_args()
    logging_utils.configure(args)

    logger.info(args)

    kwargs = dict(
        model_path=args.model,
        trials_per_relation=args.n_trial,
    )
    if args.relation is not None:
        kwargs["relation_names"] = [args.relation]

    run_ablation_experiment(**kwargs)
