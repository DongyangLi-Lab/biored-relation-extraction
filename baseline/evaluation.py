"""Shared evaluation and output writing for both tasks.

Q1: precision, recall, F1 for the has-relation class, average precision, and the same scores split by same_sentence.
Q2: eight-class macro-F1, per-class F1 and support, and a confusion matrix in RELATION_TYPES order.
"""

import json

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from common import KEY_COLUMNS, OUTPUT_DIR, RELATION_TYPES


def pick_threshold(y_true, scores):
    """Return the score threshold that maximizes has-relation F1. Call on Dev only."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # The last precision/recall point has no threshold.
    return float(thresholds[np.argmax(f1[:-1])])


def binary_scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"n": int(len(y_true)), "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def q1_metrics(table, scores, predicted):
    y = table["annotated_relation"].to_numpy()
    result = binary_scores(y, predicted)
    result["average_precision"] = round(float(average_precision_score(y, scores)), 4)
    result["by_same_sentence"] = {}
    for value, name in [(1, "same_sentence"), (0, "different_sentences")]:
        mask = (table["same_sentence"] == value).to_numpy()
        result["by_same_sentence"][name] = binary_scores(y[mask], predicted[mask])
    return result


def q2_metrics(table, predicted):
    y = table["relation_type"].to_numpy()
    p, r, f, support = precision_recall_fscore_support(
        y, predicted, labels=RELATION_TYPES, zero_division=0
    )
    present = [label for label, n in zip(RELATION_TYPES, support) if n > 0]
    return {
        "n": int(len(y)),
        "macro_f1_8_classes": round(float(f1_score(y, predicted, labels=RELATION_TYPES, average="macro", zero_division=0)), 4),
        # Dev has no Drug_Interaction or Conversion, so this version averages only the classes that appear.
        "macro_f1_present_classes": round(float(f1_score(y, predicted, labels=present, average="macro", zero_division=0)), 4),
        "per_class": {
            label: {"n": int(n), "precision": round(pi, 4), "recall": round(ri, 4), "f1": round(fi, 4)}
            for label, pi, ri, fi, n in zip(RELATION_TYPES, p, r, f, support)
        },
        "confusion_matrix": {
            "labels": RELATION_TYPES,
            "rows_true_columns_predicted": confusion_matrix(y, predicted, labels=RELATION_TYPES).tolist(),
        },
    }


def save_predictions(task, model, split, table, predicted, scores=None):
    out = table[KEY_COLUMNS].copy()
    if task == "q1":
        out["annotated_relation"] = table["annotated_relation"].to_numpy()
        out["score"] = scores
    else:
        out["relation_type"] = table["relation_type"].to_numpy()
    out["predicted"] = predicted
    path = OUTPUT_DIR / "predictions" / f"{task}_{model}_{split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def save_results(task, model, results):
    path = OUTPUT_DIR / "results" / f"{task}_{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def print_q1(split, m):
    print(f"{split}: n={m['n']}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  AP={m['average_precision']:.3f}")
    for name, g in m["by_same_sentence"].items():
        print(f"    {name:<20} n={g['n']:<5}  P={g['precision']:.3f}  R={g['recall']:.3f}  F1={g['f1']:.3f}")


def print_q2(split, m):
    print(f"{split}: n={m['n']}  macro-F1 (8 classes)={m['macro_f1_8_classes']:.3f}  "
          f"macro-F1 (classes present)={m['macro_f1_present_classes']:.3f}")
    for label, c in m["per_class"].items():
        print(f"    {label:<22} n={c['n']:<4}  F1={c['f1']:.3f}")


def add_test_flag(parser):
    parser.add_argument(
        "--test",
        action="store_true",
        help="Also evaluate on Test. Use only after the method is fixed on Dev.",
    )
    return parser


def evaluated_splits(args):
    return ["Dev", "Test"] if args.test else ["Dev"]
