import json
import logging
import pickle
from itertools import chain
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from src.dataset.rome_dataclasses import AttributeSnippets
from src.globals import *

logger = logging.getLogger(__name__)

# REMOTE_IDF_URL = f"{REMOTE_ROOT_URL}/data/dsets/idf.npy"
# REMOTE_VOCAB_URL = f"{REMOTE_ROOT_URL}/data/dsets/tfidf_vocab.json"

REMOTE_TFIDF_URL = f"{REMOTE_ROOT_URL}/data/dsets/tfidf_vectorizer.pkl"


def get_tfidf_vectorizer(data_dir: str):
    """
    Returns an sklearn TF-IDF vectorizer. See their website for docs.
    Loading hack inspired by some online blog post lol.
    """

    data_dir = Path(data_dir)

    tfidf_loc = data_dir / "tfidf_vectorizer.pkl"
    if not (tfidf_loc.exists()):
        logger.info(f"TFIDF vectorizer not found locally {tfidf_loc}")
        collect_stats(data_dir)

    with open(tfidf_loc, "rb") as f:
        vec = pickle.load(f)

    return vec


def collect_stats(data_dir: str):
    """
    Uses wikipedia snippets to collect statistics over a corpus of English text.
    Retrieved later when computing TF-IDF vectors.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    tfidf_loc = data_dir / "tfidf_vectorizer.pkl"

    try:
        print(f"Downloading IDF cache from {REMOTE_TFIDF_URL}")
        torch.hub.download_url_to_file(REMOTE_TFIDF_URL, tfidf_loc, progress=True)
        return
    except Exception as e:
        print(f"Error downloading file:", e)
        print("Recomputing TF-IDF stats...")

    snips_list = AttributeSnippets(data_dir).snippets_list
    documents = list(chain(*[[y["text"] for y in x["samples"]] for x in snips_list]))

    vec = TfidfVectorizer()
    vec.fit(documents)

    with open(tfidf_loc, "wb") as f:
        pickle.dump(vec, f)
