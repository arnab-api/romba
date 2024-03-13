import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import models
from src.globals import *
from src.rome_utils.nethook import Trace, set_requires_grad
from src.rome_utils.runningstats import (
    CombinedStat,
    Mean,
    NormMean,
    SecondMoment,
    tally,
)
from src.utils import logging_utils

logger = logging.getLogger(__name__)

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--model_name",
        default="gpt2-xl",
        choices=["gpt2-xl", "EleutherAI/gpt-j-6B", "state-spaces/mamba-2.8b-slimpj"],
    )
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    logging_utils.configure(args)
    logger.info(args)

    mt = models.ModelandTokenizer(
        model_path=args.model_name,
        torch_dtype=torch.float32,
    )
    set_requires_grad(False, mt.model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )

        if models.is_mamba_variant(mt):
            # TODO(arnab): This will change for different hooks inside the MambaBlock
            layer_name = mt.layer_name_format.format(layer_num) + ".mixer.out_proj"
        else:
            proj_layer_name = "c_proj" if "gpt2" in mt.name.lower() else "fc_out"
            layer_name = (
                mt.mlp_module_name_format.format(layer_num) + f".{proj_layer_name}"
            )

        layer_stats(
            mt,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    mt: models.ModelandTokenizer,
    layer_name: str,
    stats_dir: str,
    ds_name,
    to_collect,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],
        )
        maxlen = model.config.n_positions if hasattr(model, "config") else 2048
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    project_root = Path(__file__).parent.parent.parent
    stats_dir = os.path.join(project_root, stats_dir)

    model, tokenizer = mt.model, mt.tokenizer
    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    npos = model.config.n_positions if hasattr(model, "config") else 2048
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix

    model_name = mt.name.lower().replace("/", "_")

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    logger.info(f"searching for cached stats in => {filename}")

    if not filename.exists() and download:
        logger.info(f"stats not found locally.")
        remote_url = f"{REMOTE_ROOT_URL}/data/stats/{file_extension}"
        try:
            logger.info(f"Attempting to download {file_extension} from {remote_url}.")
            (stats_dir / "/".join(file_extension.split("/")[:-1])).mkdir(
                exist_ok=True, parents=True
            )
            torch.hub.download_url_to_file(remote_url, filename)
            logger.info("Successfully downloaded.")
        except Exception as e:
            logger.error(f"Unable to download due to {e}. Computing locally....")

    ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                # TODO(arnab): Will not be so straight-forward for different hooks inside the MambaBlock
                # Will only work for an explicit Module.
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    mt(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat


if __name__ == "__main__":
    main()
