"""Q2 baseline: always predict the most frequent relation type in Train.

Usage (from repository root):
    python baseline/q2_majority.py
    python baseline/q2_majority.py --test
"""

import argparse

import numpy as np

from common import ROOT, load_q2
from evaluation import add_test_flag, evaluated_splits, print_q2, q2_metrics, save_predictions, save_results

MODEL = "majority"


def main():
    args = add_test_flag(argparse.ArgumentParser(description=__doc__)).parse_args()
    table = load_q2()
    majority = table.loc[table["split"] == "Train", "relation_type"].mode()[0]
    print(f"Most frequent relation type in Train: {majority}")

    results = {"majority_class": majority}
    for split in evaluated_splits(args):
        part = table[table["split"] == split]
        predicted = np.full(len(part), majority, dtype=object)
        results[split] = q2_metrics(part, predicted)
        print_q2(split, results[split])
        save_predictions("q2", MODEL, split, part, predicted)

    print(f"Saved {save_results('q2', MODEL, results).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
