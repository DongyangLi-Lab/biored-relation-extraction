"""Shared data loading and structural feature preprocessing for the baseline models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = ROOT / "data" / "candidate_pairs.csv"
LABELS = ROOT / "data" / "relation_labels.csv"
FEATURES = ROOT / "baseline" / "outputs" / "structural_features.csv"
OUTPUT_DIR = ROOT / "baseline" / "outputs"

KEY_COLUMNS = ["split", "pmid", "pair_type", "entity1", "entity2"]
COUNT_COLUMNS = ["min_word_gap", "entity1_mentions", "entity2_mentions"]

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


def read_table(path):
    # dtype=str keeps IDs such as 3175 and D003924 as identical text in every table.
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def pair_key(table):
    """split + pmid + pair_type + the two entity IDs in sorted order."""
    low = table[["entity1", "entity2"]].min(axis=1)
    high = table[["entity1", "entity2"]].max(axis=1)
    return table["split"] + "|" + table["pmid"] + "|" + table["pair_type"] + "|" + low + "|" + high


def load_q1():
    """Every candidate pair with its 0/1 label and structural features."""
    candidates = read_table(CANDIDATES)
    candidates["annotated_relation"] = candidates["annotated_relation"].astype(int)
    if not FEATURES.exists():
        raise SystemExit("Run python baseline/structural_features.py first.")
    features = read_table(FEATURES)
    for column in ["shares_sentence", *COUNT_COLUMNS]:
        features[column] = features[column].astype(int)

    candidates["key"] = pair_key(candidates)
    features["key"] = pair_key(features)
    table = candidates.merge(features.drop(columns=KEY_COLUMNS), on="key", how="left", validate="one_to_one")
    if table["shares_sentence"].isna().any():
        raise SystemExit("Some candidate pairs have no structural features. Rerun structural_features.py.")
    return table


def load_q2():
    """Annotated pairs with their relation type and the same structural features as Q1."""
    labels = read_table(LABELS)
    labels["key"] = pair_key(labels)
    q1 = load_q1()
    table = labels[["key", "relation_type"]].merge(q1, on="key", how="left", validate="one_to_one")
    if table["annotated_relation"].isna().any():
        raise SystemExit("Some labeled pairs are missing from the candidate table.")
    return table


def structural_transformer():
    """log1p + standardize counts, keep shares_sentence as 0/1, one-hot pair_type. Fitted on Train only."""
    counts = make_pipeline(FunctionTransformer(np.log1p), StandardScaler())
    return ColumnTransformer(
        [
            ("counts", counts, COUNT_COLUMNS),
            ("same_sentence", "passthrough", ["shares_sentence"]),
            ("pair_type", OneHotEncoder(handle_unknown="ignore"), ["pair_type"]),
        ]
    )
