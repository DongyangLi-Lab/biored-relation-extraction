"""Q1 classifiers: does BioRED annotate a relation between this pair?

Fits on Train, picks the decision threshold that maximizes has-relation F1 on Dev,
and applies the same threshold to Test.

Models and inputs:
    logreg  LogisticRegression(C=1)  structural, text, both
    svm     LinearSVC(C=1)           text
    nb      MultinomialNB(alpha=1)   text

    structural  same_sentence, min_token_gap, entity mention counts, pair_type
    text        TF-IDF words and two-word phrases from context_local
    both        structural and text side by side

Usage (from repository root):
    python baseline/q1_models.py --model logreg --features structural
    python baseline/q1_models.py --model svm --features text
    python baseline/q1_models.py --model nb --features text --test
"""

import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from common import FEATURE_SETS, ROOT, load_q1
from evaluation import add_test_flag, evaluated_splits, pick_threshold, print_q1, q1_metrics, save_predictions, save_results

MODELS = {
    "logreg": (lambda: LogisticRegression(solver="lbfgs", C=1, max_iter=1000), ["structural", "text", "both"]),
    "svm": (lambda: LinearSVC(C=1), ["text"]),
    # MultinomialNB needs non-negative inputs, so it only takes the TF-IDF text features.
    "nb": (lambda: MultinomialNB(alpha=1.0), ["text"]),
}


def has_relation_scores(model, table):
    """Probability of a relation for logreg and nb; signed distance from the boundary for svm."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(table)[:, 1]
    return model.decision_function(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--features", choices=FEATURE_SETS, required=True)
    args = add_test_flag(parser).parse_args()
    make_classifier, allowed = MODELS[args.model]
    if args.features not in allowed:
        parser.error(f"--model {args.model} runs with --features {' or '.join(allowed)}")
    model_name = f"{args.model}_{args.features}"

    table = load_q1()
    train = table[table["split"] == "Train"]
    model = make_pipeline(FEATURE_SETS[args.features](), make_classifier())
    model.fit(train, train["annotated_relation"])

    dev = table[table["split"] == "Dev"]
    threshold = pick_threshold(dev["annotated_relation"], has_relation_scores(model, dev))
    print(f"{model_name}: threshold chosen on Dev = {threshold:.4f}")

    # Train scores use the same Dev threshold; comparing them with Dev and Test shows how much the model overfits.
    train_scores = has_relation_scores(model, train)
    results = {
        "threshold": round(threshold, 4),
        "Train": q1_metrics(train, train_scores, (train_scores >= threshold).astype(int)),
    }
    for split in evaluated_splits(args):
        part = table[table["split"] == split]
        scores = has_relation_scores(model, part)
        predicted = (scores >= threshold).astype(int)
        results[split] = q1_metrics(part, scores, predicted)
        print_q1(split, results[split])
        save_predictions("q1", model_name, split, part, predicted, scores=scores)

    print(f"Saved {save_results('q1', model_name, results).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
