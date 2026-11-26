"""Shared data loading and structural feature preprocessing for the baseline models.

All inputs come from the shared preprocessing tables built by sota/prepare_pairs.py:
    data/processed/Train.csv, Dev.csv, Test.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "baseline" / "outputs"
SPLITS = ("Train", "Dev", "Test")

# In the shared tables, entity1 and entity2 are already in sorted order, so these columns form the shared sample key.
KEY_COLUMNS = ["split", "pmid", "pair_type", "entity1", "entity2"]
COUNT_COLUMNS = ["min_token_gap", "entity1_mentions", "entity2_mentions"]
INT_COLUMNS = ["annotated_relation", "same_sentence", *COUNT_COLUMNS]
TEXT_COLUMN = "context_local"

# Fixed order for Q2 per-class scores and confusion matrices.
RELATION_TYPES = [
    "Association",
    "Positive_Correlation",
    "Negative_Correlation",
    "Bind",
    "Cotreatment",
    "Comparison",
    "Drug_Interaction",
    "Conversion",
]


def load_q1():
    """Every candidate pair with its 0/1 label, structural features, and shared text."""
    missing = [split for split in SPLITS if not (PROCESSED_DIR / f"{split}.csv").exists()]
    if missing:
        raise SystemExit("Run python sota/prepare_pairs.py first to build data/processed/.")
    # dtype=str keeps IDs such as 3175 and D003924 as text; keep_default_na=False keeps empty relation_type as "".
    table = pd.concat(
        [pd.read_csv(PROCESSED_DIR / f"{split}.csv", dtype=str, keep_default_na=False) for split in SPLITS],
        ignore_index=True,
    )
    for column in INT_COLUMNS:
        table[column] = table[column].astype(int)
    return table


def load_q2():
    """Annotated pairs only, with their relation type."""
    table = load_q1()
    return table[table["annotated_relation"] == 1].reset_index(drop=True)


def structural_parts():
    """log1p + standardize counts, keep same_sentence as 0/1, one-hot pair_type."""
    counts = make_pipeline(FunctionTransformer(np.log1p), StandardScaler())
    return [
        ("counts", counts, COUNT_COLUMNS),
        ("same_sentence", "passthrough", ["same_sentence"]),
        ("pair_type", OneHotEncoder(handle_unknown="ignore"), ["pair_type"]),
    ]


def text_parts():
    """TF-IDF words and two-word phrases from the shared ENTITY1/ENTITY2 text.

    No stop-word list, so negation words such as "not" and "no" are kept.
    """
    return [("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), TEXT_COLUMN)]


# Every transformer is fitted inside the model pipeline, so vocabulary and scaling come from Train only.
FEATURE_SETS = {
    "structural": lambda: ColumnTransformer(structural_parts()),
    "text": lambda: ColumnTransformer(text_parts()),
    "both": lambda: ColumnTransformer(structural_parts() + text_parts()),
}
