import argparse
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import torch.utils.data
from dataclasses_json import DataClassJsonMixin
from torch.utils.data import Dataset

from src.utils.globals import *

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictedToken(DataClassJsonMixin):
    """A predicted token and its probability."""

    token: str
    prob: float

    def __str__(self) -> str:
        return f'"{self.token}" (p={self.prob:.3f})'


@dataclass(frozen=True)
class CounterFactualSample(DataClassJsonMixin):
    """A single (subject, object) pair in a relation."""

    relation_id: str  # wikidata relation id, maybe helpful for filtering
    prompt: str
    subject: str
    target_true: str
    target_edit: str
    evaluation_prompts: dict[str, list[str]] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.prompt} [{self.target_true} -> {self.target_edit}]"


REMOTE_ROOT = f"{REMOTE_ROOT_URL}/data/dsets"


@dataclass()
class CounterFactDataset(Dataset, DataClassJsonMixin):
    data: list[CounterFactualSample]

    def __init__(
        self,
        data_dir: str,
        multi: bool = False,
        size: Optional[int] = None,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        cf_loc = data_dir / (
            "counterfact.json" if not multi else "multi_counterfact.json"
        )
        if not cf_loc.exists():
            remote_url = f"{REMOTE_ROOT}/{'multi_' if multi else ''}counterfact.json"
            logging.info(f"{cf_loc} does not exist. Downloading from {remote_url}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(remote_url, cf_loc)

        with open(cf_loc, "r") as f:
            json_file = json.load(f)
        size = size if size is not None else len(json_file)

        self.data = [
            CounterFactualSample(
                relation_id=sample["requested_rewrite"]["relation_id"],
                prompt=sample["requested_rewrite"]["prompt"].format(
                    sample["requested_rewrite"]["subject"]
                ),
                subject=sample["requested_rewrite"]["subject"],
                target_true=sample["requested_rewrite"]["target_true"]["str"],
                target_edit=sample["requested_rewrite"]["target_new"]["str"],
                evaluation_prompts={
                    eval_type: sample[eval_type]
                    for eval_type in [
                        "paraphrase_prompts",
                        "neighborhood_prompts",
                        "generation_prompts",
                    ]
                },
            )
            for sample in json_file[:size]
        ]

        logging.info(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MultiCounterFactDataset(CounterFactDataset):
    def __init__(self, data_dir: str, size: Optional[int] = None, *args, **kwargs):
        super().__init__(data_dir, *args, multi=True, size=size, **kwargs)


# ------------------------------------------
# experiment dataclasses
# ------------------------------------------

# @dataclass(frozen=True)
# class SampleResult(DataClassJsonMixin):
#     query: str
#     answer: str
#     prediction: list[PredictedToken]


# @dataclass(frozen=True)
# class LayerResult(DataClassJsonMixin):
#     samples: list[SampleResult]
#     score: float


# @dataclass(frozen=True)
# class TrialResult(DataClassJsonMixin):
#     few_shot_demonstration: str
#     faithfulness: dict[Layer, LayerResult]
#     # efficacy: dict[Layer, float] # TODO: may add this later


# @dataclass(frozen=True)
# class ExperimentResults(DataClassJsonMixin):
#     experiment_specific_args: dict[str, Any]
#     trial_results: list[TrialResult]


# @dataclass(frozen=True)
# class ReprReplacementResults(DataClassJsonMixin):
#     source_QA: Sample
#     edit_QA: Sample
#     edit_index: int
#     edit_token: str
#     predictions_after_patching: dict[int, list[PredictedToken]]
#     rank_edit_ans_after_patching: dict[int, int]


# @dataclass(frozen=True)
# class RelationProperties(DataClassJsonMixin):
#     """Some metadata about a relation."""

#     relation_type: str
#     domain_name: str
#     range_name: str
#     symmetric: bool
#     fn_type: str
#     # disambiguating: bool
