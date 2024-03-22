import argparse
import json
import logging
import os
from typing import Optional

from src.dataset.dataclasses import load_relation
from src.dataset.rome_dataclasses import CounterFactDataset
from src.functional import filter_samples_by_model_knowledge
from src.globals import DATA_DIR, RESULTS_DIR
from src.models import ModelandTokenizer
from src.tracing import calculate_average_indirect_effects
from src.utils import logging_utils

logger = logging.getLogger(__name__)


def causal_trace_relations(
    model_path: str,
    trials_per_relation: Optional[int] = None,
    relation_names: list[str] = [
        "place_in_city",
        "country_capital_city",
        "person_occupation",
        "person_plays_pro_sport",
        "company_ceo",
        "company_hq",
        "landmark_in_country",
        "product_by_company",
    ],
    hook: Optional[str] = None,
):
    mt = ModelandTokenizer(model_path=model_path)
    relation_dir = os.path.join(DATA_DIR, "relation", "factual")
    hook_name = hook if hook is not None else "residual"
    results_dir = os.path.join(RESULTS_DIR, "causal_tracing_aie", hook_name)
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
            limit=trials_per_relation if trials_per_relation > 50 else None,
        )

        relation_save_path = os.path.join(results_dir, f"{relation_name}.json")
        avg_indirect_effect = calculate_average_indirect_effects(
            mt=mt,
            prompt=relation.prompt_templates[0],
            samples=relation.samples,
            corruption_strategy="alt_patch",
            n_trials=trials_per_relation,
            save_path=relation_save_path,
            verbose=True,
            mamba_block_hook=hook,
        )
        logger.info("Saved results to => " + relation_save_path)
        logger.info("-" * 50)


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
    parser.add_argument(
        "--hook",
        type=str,
        default=None,
        choices=["ssm_after_ssm", "mlp_after_silu", "after_down_proj"],
    )

    args = parser.parse_args()
    logging_utils.configure(args)

    logger.info(args)

    kwargs = dict(
        model_path=args.model,
        trials_per_relation=args.n_trial,
        hook=args.hook,
    )
    if args.relation is not None:
        kwargs["relation_names"] = [args.relation]

    causal_trace_relations(**kwargs)
