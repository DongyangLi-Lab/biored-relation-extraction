"""Q1 baseline: same-sentence co-occurrence rule.

Predicts a relation when two entities are mentioned in at least one shared sentence,
using the shared same_sentence column.

Usage (from repository root):
    python sota/prepare_pairs.py                 # once, to build data/processed/
    python baseline/same_sentence_rule.py        # Dev
    python baseline/same_sentence_rule.py --test # Dev and Test, once the method is fixed
"""

import argparse

from common import ROOT, load_q1
from evaluation import add_test_flag, evaluated_splits, print_q1, q1_metrics, save_predictions, save_results

MODEL = "same_sentence_rule"


def main():
    args = add_test_flag(argparse.ArgumentParser(description=__doc__)).parse_args()
    table = load_q1()

    results = {}
    for split in evaluated_splits(args):
        part = table[table["split"] == split]
        predicted = part["same_sentence"].to_numpy()
        # The rule has no graded score, so the 0/1 prediction doubles as the score for average precision.
        results[split] = q1_metrics(part, predicted, predicted)
        print_q1(split, results[split])
        save_predictions("q1", MODEL, split, part, predicted, scores=predicted)

    print(f"Saved {save_results('q1', MODEL, results).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
