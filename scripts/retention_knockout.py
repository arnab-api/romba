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
import src.tokens as tokenization_utils

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
from src.tracing import calculate_average_indirect_effects, detensorize_indirect_effects
from src.utils import logging_utils

logger = logging.getLogger(__name__)


def load_mean_activations(
    mt: ModelandTokenizer,
    num_docs=128,
    n_tok_per_doc=128,
):
    ACT_DIR = os.path.join(DATA_DIR, "mean_activations")
    os.makedirs(ACT_DIR, exist_ok=True)
    FILE_NAME = mt.name.lower().split("/")[-1] + ".json"
    if FILE_NAME in os.listdir(ACT_DIR):
        logger.info("Loading mean activations from cache")
        with open(os.path.join(ACT_DIR, FILE_NAME), "r") as f:
            mean_activations = json.load(f)
        for layer in mean_activations:
            for hook in mean_activations[layer]:
                mean_activations[layer][hook] = torch.tensor(
                    mean_activations[layer][hook]
                ).to(mt.device)
        return mean_activations

    logger.info("Calculating mean activations")

    with open(os.path.join(DATA_DIR, "attribute_snippets.json"), "r") as f:
        attribute_snippets = json.load(f)

    random_text = [
        attribute_snippets[i]["samples"][0]["text"]
        for i in range(min(len(attribute_snippets), num_docs))
    ]

    hooks = [
        "ssm_after_up_proj",
        "ssm_after_conv1D",
        "ssm_after_silu",
        "ssm_after_ssm",
        "mlp_after_up_proj",
        "mlp_after_silu",
        "before_down_proj",
        "after_down_proj",  # the output of the mamba block #! Not the residual
    ]

    avg_activations = {
        layer: {hook: None for hook in hooks} for layer in mt.layer_names
    }

    for text in tqdm(random_text):
        inputs = mt.tokenizer(
            text,
            return_tensors="pt",
        ).to(mt.device)
        input_ids = inputs["input_ids"]
        input_ids = input_ids[:, : min(input_ids.shape[-1], n_tok_per_doc)]

        mt.reset_forward()

        current_states = {
            layer: {hook: None for hook in hooks} for layer in mt.layer_names
        }
        for layer in mt.layer_names:
            mambablock = baukit.get_module(mt.model, name=layer + ".mixer")
            mambablock.forward = types.MethodType(
                MambaBlockForwardPatcher(retainer=current_states[layer]), mambablock
            )

        with torch.no_grad():
            mt.model(input_ids)

        for layer in mt.layer_names:
            for hook in hooks:
                activations = current_states[layer][hook]
                # print(activations.shape)
                if avg_activations[layer][hook] is None:
                    avg_activations[layer][hook] = activations
                else:
                    avg_activations[layer][hook] = torch.cat(
                        (avg_activations[layer][hook], activations), dim=1
                    )

        functional.free_gpu_cache()

    for layer in avg_activations:
        for hook in avg_activations[layer]:
            avg_activations[layer][hook] = avg_activations[layer][hook].mean(dim=1)

    for hook in hooks:
        logger.info(hook, avg_activations["layers.4"][hook].shape)

    avg_detensorized = {}
    for layer in avg_activations:
        avg_detensorized[layer] = detensorize_indirect_effects(avg_activations[layer])

    with open(os.path.join(ACT_DIR, FILE_NAME), "w") as f:
        json.dump(avg_activations, f)
        logger.info(f"Mean activations saved to {os.path.join(ACT_DIR, FILE_NAME)}")

    return avg_activations


def get_window(layer_idx, num_layers=64, window_size=10):
    window_size = window_size // 2
    start = max(0, layer_idx - window_size)
    end = min(num_layers, layer_idx + window_size)
    return list(range(start, end))


def retention_knockout_on_single_fact(
    mt: ModelandTokenizer,
    subject: str,
    prompt_template: str,
    mean_activations: dict,
    patch_hook: str = "ssm_after_up_proj",
    window=10,
):
    prompt = tokenization_utils.maybe_prefix_eos(mt, prompt_template.format(subject))
    inputs = mt.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True).to(
        mt.device
    )
    offsets = inputs.pop("offset_mapping")[0]

    e_range = functional.find_token_range(
        string=prompt,
        substring=subject,
        offset_mapping=offsets,
    )

    logger.debug(f"Subject range: {e_range}")

    prompt_last = inputs.input_ids.shape[-1] - 1
    subj_positions = list(range(e_range[0], e_range[1]))
    non_subj_positions = [
        i
        for i in range(inputs.input_ids.shape[-1])
        if i not in subj_positions + [prompt_last]
    ]

    # caching states from the clean run
    mt.reset_forward()
    clean_states = {layer: {patch_hook: None} for layer in mt.layer_names}
    for layer in mt.layer_names:
        mambablock = baukit.get_module(mt.model, name=layer + ".mixer")
        mambablock.forward = types.MethodType(
            MambaBlockForwardPatcher(retainer=clean_states[layer]), mambablock
        )

    output_clean = mt(**inputs)
    proba = torch.nn.functional.softmax(output_clean[:, -1], dim=-1)
    ans_t = proba.argmax(dim=-1)
    ans = mt.tokenizer.decode(ans_t)
    p_ans = proba[0, ans_t].item()

    logger.info(f"{subject} -> {ans} ({p_ans})")

    ablate_positions = {
        "subject": subj_positions,
        "subj_last": [subj_positions[-1]],
        "non_subject": non_subj_positions,
        "prompt_last": [prompt_last],
    }
    result = {"answer": ans, "p_answer": p_ans, "knock_out_from_last": {}}

    for setting, ablate_position in ablate_positions.items():
        corrupted_states = {}
        patch_hook = "ssm_after_up_proj"
        restore_positions = list(
            set(range(prompt_last)) - set(ablate_position)
        )  # Don't restore prompt_last (let the model calculate this)

        layer_wise_p_ans = []
        for layer_idx in range(mt.n_layer):
            # corrupted run with mean ablation
            mt.reset_forward()

            current_window = get_window(
                layer_idx, num_layers=mt.n_layer, window_size=window
            )

            for l in current_window:
                layername = mt.layer_name_format.format(l)
                mambablock = baukit.get_module(mt.model, name=layername + ".mixer")
                patch_spec = {}
                for i in ablate_position:
                    patch_spec[i] = mean_activations[layername][patch_hook]
                for i in restore_positions:
                    patch_spec[i] = clean_states[layername][patch_hook][:, i, :]

                mambablock.forward = types.MethodType(
                    MambaBlockForwardPatcher(
                        patch_spec=patch_spec,
                        patch_hook=patch_hook,
                        retainer=corrupted_states,
                    ),
                    mambablock,
                )
            # print(inputs)
            output_corrupted = mt(**inputs)
            proba_corrupted = torch.nn.functional.softmax(
                output_corrupted[:, -1], dim=-1
            )
            p_ans_corrupted = proba_corrupted[0, ans_t].item()
            layer_wise_p_ans.append(p_ans_corrupted)

            functional.free_gpu_cache()

        logger.info(f"{setting} -> {layer_wise_p_ans}")
        result["knock_out_from_last"][setting] = layer_wise_p_ans

    mt.reset_forward()
    return result


def run_retention_knockout(
    model_path: str,
    trials_per_relation: Optional[int] = None,
    relation_names: list[str] = [
        "place_in_city",
        "country_capital_city",
        "person_occupation",
        "person_plays_pro_sport",
        "company_hq",
        "landmark_in_country",
        "product_by_company",
    ],
):
    mt = ModelandTokenizer(model_path=model_path)
    relation_dir = os.path.join(DATA_DIR, "relation", "factual")
    results_dir = os.path.join(RESULTS_DIR, "knockout")
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to {results_dir}")

    mean_activations = load_mean_activations(mt)

    for relation_name in relation_names:
        logger.info("-" * 50)
        logger.info(f"Processing relation => {relation_name}")
        relation = load_relation(
            file=os.path.join(relation_dir, f"{relation_name}.json")
        )
        relation.select_icl_examples(0)
        relation = filter_samples_by_model_knowledge(
            mt=mt, relation=relation, limit=trials_per_relation
        )

        knock_out_results = []
        for sample in tqdm(relation.samples):
            knock_out_results.append(
                retention_knockout_on_single_fact(
                    mt=mt,
                    subject=sample.subject,
                    prompt_template=relation.prompt_templates[0],
                    mean_activations=mean_activations,
                )
            )
            functional.free_gpu_cache()

        with open(os.path.join(results_dir, f"{relation_name}.json"), "w") as f:
            json.dump(knock_out_results, f)
            logger.info(
                f"Results saved to {os.path.join(results_dir, f'{relation_name}.json')}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging_utils.add_logging_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "state-spaces/mamba-2.8b",
            # "EleutherAI/pythia-2.8b-deduped", # ! doesn't support yet. As a quick fix, just copy the original implementation from ROME repo
        ],
        default="state-spaces/mamba-2.8b",
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

    run_retention_knockout(**kwargs)
