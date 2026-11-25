"""Q2 classifiers: which of the eight relation types applies?

Fits on Train (annotated pairs only) and predicts one of the eight types directly.

Models and inputs:
    logreg  multinomial LogisticRegression(C=1)  structural, text, both
    nb      MultinomialNB(alpha=1)               text

    structural  same_sentence, min_token_gap, entity mention counts, pair_type
    text        TF-IDF words and two-word phrases from context_local
    both        structural and text side by side

Usage (from repository root):
    python baseline/q2_models.py --model logreg --features structural
    python baseline/q2_models.py --model nb --features text --test
"""

import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from common import FEATURE_SETS, ROOT, load_q2
from evaluation import add_test_flag, evaluated_splits, print_q2, q2_metrics, save_predictions, save_results

MODELS = {
    # With lbfgs and more than two classes, scikit-learn fits a single multinomial model.
    "logreg": (lambda: LogisticRegression(solver="lbfgs", C=1, max_iter=1000), ["structural", "text", "both"]),
    # MultinomialNB needs non-negative inputs, so it only takes the TF-IDF text features.
    "nb": (lambda: MultinomialNB(alpha=1.0), ["text"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--features", choices=FEATURE_SETS, required=True)
    args = add_test_flag(parser).parse_args()
    make_classifier, allowed = MODELS[args.model]
    if args.features not in allowed:
        parser.error(f"--model {args.model} runs with --features {' or '.join(allowed)}")
    model_name = f"{args.model}_{args.features}"

    table = load_q2()
    train = table[table["split"] == "Train"]
    model = make_pipeline(FEATURE_SETS[args.features](), make_classifier())
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
