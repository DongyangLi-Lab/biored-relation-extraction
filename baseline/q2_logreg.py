"""Q2 multinomial logistic regression: which of the eight relation types applies?

Fits on Train (annotated pairs only) and predicts one of the eight types directly.

Inputs:
    structural  shares_sentence, min_word_gap, entity mention counts, pair_type
    (text and structural+text are added once the shared ENTITY1/ENTITY2 text column is available)

Usage (from repository root):
    python baseline/q2_logreg.py --features structural
    python baseline/q2_logreg.py --features structural --test
"""

import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import ROOT, load_q2, structural_transformer
from evaluation import add_test_flag, evaluated_splits, print_q2, q2_metrics, save_predictions, save_results

FEATURE_SETS = {"structural": structural_transformer}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", choices=FEATURE_SETS, required=True)
    args = add_test_flag(parser).parse_args()
    model_name = f"logreg_{args.features}"

    table = load_q2()
    train = table[table["split"] == "Train"]
    # With lbfgs and more than two classes, scikit-learn fits a single multinomial model.
    model = make_pipeline(
        FEATURE_SETS[args.features](),
        LogisticRegression(solver="lbfgs", C=1, max_iter=1000),
    )
    model.fit(train, train["relation_type"])

    results = {}
    for split in evaluated_splits(args):
        part = table[table["split"] == split]
        predicted = model.predict(part)
        results[split] = q2_metrics(part, predicted)
        print_q2(split, results[split])
        save_predictions("q2", model_name, split, part, predicted)

    print(f"Saved {save_results('q2', model_name, results).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
