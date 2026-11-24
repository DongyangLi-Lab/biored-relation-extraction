"""Q1 logistic regression: does BioRED annotate a relation between this pair?

Fits on Train, picks the decision threshold that maximizes has-relation F1 on Dev,
and applies the same threshold to Test.

Inputs:
    structural  shares_sentence, min_word_gap, entity mention counts, pair_type
    (text and structural+text are added once the shared ENTITY1/ENTITY2 text column is available)

Usage (from repository root):
    python baseline/q1_logreg.py --features structural
    python baseline/q1_logreg.py --features structural --test
"""

import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import ROOT, load_q1, structural_transformer
from evaluation import add_test_flag, evaluated_splits, pick_threshold, print_q1, q1_metrics, save_predictions, save_results

FEATURE_SETS = {"structural": structural_transformer}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", choices=FEATURE_SETS, required=True)
    args = add_test_flag(parser).parse_args()
    model_name = f"logreg_{args.features}"

    table = load_q1()
    train = table[table["split"] == "Train"]
    model = make_pipeline(
        FEATURE_SETS[args.features](),
        LogisticRegression(solver="lbfgs", C=1, max_iter=1000),
    )
    model.fit(train, train["annotated_relation"])

    dev = table[table["split"] == "Dev"]
    threshold = pick_threshold(dev["annotated_relation"], model.predict_proba(dev)[:, 1])
    print(f"Threshold chosen on Dev: {threshold:.4f}")

    results = {"threshold": round(threshold, 4)}
    for split in evaluated_splits(args):
        part = table[table["split"] == split]
        scores = model.predict_proba(part)[:, 1]
        predicted = (scores >= threshold).astype(int)
        results[split] = q1_metrics(part, scores, predicted)
        print_q1(split, results[split])
        save_predictions("q1", model_name, split, part, predicted, scores=scores)

    print(f"Saved {save_results('q1', model_name, results).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
