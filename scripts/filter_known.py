import argparse
import json
import logging
import os
from typing import Optional

from src.dataset.rome_dataclasses import CounterFactDataset
from src.functional import filter_counterfact_samples_by_model_knowledge
from src.globals import DATA_DIR
from src.models import ModelandTokenizer
from src.utils import logging_utils

logger = logging.getLogger(__name__)


def filter_known(
    model_path: str,
    limit: Optional[int] = None,
):
    mt = ModelandTokenizer(model_path=model_path)
    dataset = CounterFactDataset(DATA_DIR)

    filtered = filter_counterfact_samples_by_model_knowledge(
        mt=mt,
        counterfact=dataset,
        limit=limit,
    )

    known_dir = os.path.join(DATA_DIR, "known")
    os.makedirs(known_dir, exist_ok=True)

    save_path = os.path.join(known_dir, f"{model_path.split('/')[-1]}.json")

    with open(save_path, "w") as f:
        json.dump(filtered, f)

    logger.info(f"Saved filtered dataset to {save_path}")


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
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    logging_utils.configure(args)

    logger.info(args)

    filter_known(model_path=args.model, limit=args.limit)
