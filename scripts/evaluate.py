import json
import logging
import shutil
from itertools import islice
from pathlib import Path
from time import time
from typing import Optional, Tuple, Union

import torch

from scripts.py.eval_utils_counterfact import compute_rewrite_quality_counterfact

# from scripts.py.eval_utils_zsre import compute_rewrite_quality_zsre
from src import functional, models
from src.dataset.rome_dataclasses import (  # MENDQADataset,; MultiCounterFactDataset,
    AttributeSnippets,
    CounterFactDataset,
    get_tfidf_vectorizer,
)
from src.globals import DATA_DIR, HPARAMS_DIR, KV_DIR, RESULTS_DIR

# from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models import ModelandTokenizer
from src.rome import ROMEHyperParams, apply_rome_to_model
from src.rome.rome_main import restore_weights, save_weights
from src.rome_utils import nethook
from src.utils import experiment_utils, logging_utils

logger = logging.getLogger(__name__)

# from baselines.ft import FTHyperParams, apply_ft_to_model
# from baselines.mend import MENDHyperParams, MendRewriteExecutor


ALG_DICT = {
    # "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    # "FT": (FTHyperParams, apply_ft_to_model),
    # "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    # "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    # "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def main(
    alg_name: str,
    model_name: Union[str, ModelandTokenizer],
    hparams_fname: str,
    ds_name: str,
    dir_name: str,
    datafile: Optional[str] = None,
    dataset_size_limit: Optional[int] = None,
    continue_from_run: Optional[str] = None,
    skip_generation_tests: Optional[bool] = False,
    generation_test_interval: int = 1,
    conserve_memory: bool = True,
    num_edits: int = 1,
    use_cache: bool = False,
    layer: Optional[int] = None,
    rewrite_module_tmp: Optional[str] = None,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # parse the hparams and determine the directory to save the results
    if continue_from_run is None or not (run_dir := DIR / continue_from_run).exists():
        continue_from_run = None

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if layer is not None:
        if continue_from_run is not None:
            assert (
                layer == hparams.layers[0]
            ), f"Layer mismatch: specified layer {layer} is different from the previous run {hparams.layers[0]}"
        logger.warn(f"Overriding hparams layer to {layer}")
        hparams.layers = [layer]
    if rewrite_module_tmp is not None:
        if continue_from_run is not None:
            assert (
                rewrite_module_tmp == hparams.rewrite_module_tmp
            ), f"Rewrite module mismatch: specified rewrite module {rewrite_module_tmp} is different from the previous run {hparams.rewrite_module_tmp}"
        logger.warn(f"Overriding hparams rewrite_module_tmp to {rewrite_module_tmp}")
        hparams.rewrite_module_tmp = rewrite_module_tmp

    if continue_from_run is None:
        DIR = (
            RESULTS_DIR
            / dir_name
            / hparams.rewrite_module_tmp.split(".")[-1]
            / f"layer_{hparams.layers[0]}"
        )
        if DIR.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in DIR.iterdir()
                if str(x).split("_")[-1].isnumeric()  # and str(x).startswith("run_")
            ]
            # print("id_list: ", id_list)
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = DIR / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be stored at {run_dir}")

    # raise NotImplementedError("Need to fix the following code")

    if not (run_dir / "params.json").exists():
        # shutil.copyfile(params_path, run_dir / "params.json")
        with open(run_dir / "params.json", "w") as f:
            json.dump(hparams.__dict__, f, indent=1)
    logger.info(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        mt = ModelandTokenizer(model_path=model_name)
    else:
        mt = model_name
        model_name = mt.name

    # Load data
    logger.info("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    if datafile is None:
        ds = ds_class(DATA_DIR, tok=mt.tokenizer, size=dataset_size_limit)
    else:
        ds = ds_class(
            data_dir=datafile,
            size=dataset_size_limit,
            absolute_path=True,
            tok=mt.tokenizer,
        )

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_v_dir_name = hparams.rewrite_module_tmp.split(".")[-1]
        if hparams.mamba_block_non_ssm:
            cache_v_dir_name += "_non_ssm"
        if hparams.mamba_block_ssm:
            cache_v_dir_name += "_ssm"
        cache_template = (
            KV_DIR
            / f"{model_name.lower().replace('/', '_')}"
            / f"{alg_name}"
            / f"{cache_v_dir_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        logger.info(f"Will load cache from {cache_template}")

    # Iterate through dataset
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = (
            dict(cache_template=cache_template)
            if any(alg in alg_name for alg in ["ROME", "MEMIT"])
            else dict()
        )

        start = time()
        edited_model, original_weights = apply_algo(
            mt=mt,
            requests=[
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams=hparams,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args,
        )
        exec_time = time() - start
        print("Execution took", exec_time)

        # Evaluate new model
        start = time()
        gen_test_vars = [snips, vec]
        for record in record_chunks:
            out_file = Path(case_result_template.format(num_edits, record["case_id"]))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue

            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(
                    mt,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                ),
            }

            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)

        # Restore original weights
        restore_weights(mt.model, original_weights)

        print("Evaluation took", time() - start)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=[
            "state-spaces/mamba-2.8b",
            "EleutherAI/pythia-2.8b-deduped",
            # "gpt2-medium",
            # "gpt2-large",
            # "gpt2-xl",
            # "EleutherAI/gpt-j-6B",
        ],
        default="state-spaces/mamba-2.8b-slimpj",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="mamba-3b.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )

    parser.add_argument(
        "--datafile",
        type=str,
        default=None,
        help="Absolute path to the data file. If None, will use the default data file.",
    )

    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="layer to edit. if -1, then will parse layer from the default hparams file",
    )
    parser.add_argument(
        "--rewrite_module_tmp",
        type=str,
        default=None,
        help="the module to use for rewriting. if None, then will parse from the default hparams file",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    logging_utils.configure(args)
    logger.info(f"Running with args: {args}")

    main(
        alg_name=args.alg_name,
        model_name=args.model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        datafile=args.datafile,
        dataset_size_limit=args.dataset_size_limit,
        continue_from_run=args.continue_from_run,
        skip_generation_tests=args.skip_generation_tests,
        generation_test_interval=args.generation_test_interval,
        conserve_memory=args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        layer=args.layer if args.layer != -1 else None,
        rewrite_module_tmp=args.rewrite_module_tmp,
    )
