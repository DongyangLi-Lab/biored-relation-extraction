"""Collect baseline/outputs/results/*.json into baseline/RESULTS.md.

Usage (from repository root, after running every model with --test):
    python baseline/make_results_table.py
"""

import json

import pandas as pd

from common import OUTPUT_DIR, RELATION_TYPES, ROOT

RESULTS_DIR = OUTPUT_DIR / "results"
TARGET = ROOT / "baseline" / "RESULTS.md"

Q1_ROWS = [
    ("same_sentence_rule", "同句规则", "—"),
    ("logreg_structural", "逻辑回归", "仅结构"),
    ("logreg_text", "逻辑回归", "仅文本"),
    ("logreg_both", "逻辑回归", "结构加文本"),
    ("svm_text", "线性 SVM", "仅文本"),
    ("nb_text", "多项朴素贝叶斯", "仅文本"),
]
Q2_ROWS = [
    ("majority", "多数类", "—"),
    ("logreg_structural", "逻辑回归", "仅结构"),
    ("logreg_text", "逻辑回归", "仅文本"),
    ("logreg_both", "逻辑回归", "结构加文本"),
    ("nb_text", "多项朴素贝叶斯", "仅文本"),
]
SHORT = {
    "Association": "Assoc",
    "Positive_Correlation": "Pos",
    "Negative_Correlation": "Neg",
    "Bind": "Bind",
    "Cotreatment": "Cotreat",
    "Comparison": "Compar",
    "Drug_Interaction": "DrugInt",
    "Conversion": "Conver",
}


def load(task, model):
    return json.loads((RESULTS_DIR / f"{task}_{model}.json").read_text(encoding="utf-8"))


def table(header, rows, text_columns=2):
    """Markdown table; the first text_columns columns are left-aligned, the rest right-aligned."""
    align = ["---" if i < text_columns else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def f(x):
    return f"{x:.3f}"


def q1_section():
    r = {model: load("q1", model) for model, _, _ in Q1_ROWS}
    main_rows = [
        [name, inputs, f(r[m]["Test"]["precision"]), f(r[m]["Test"]["recall"]), f(r[m]["Test"]["f1"]),
         f(r[m]["Test"]["average_precision"]), f(r[m]["Dev"]["f1"])]
        for m, name, inputs in Q1_ROWS
    ]
    groups = r["logreg_both"]["Test"]["by_same_sentence"]
    group_rows = []
    for m, name, inputs in Q1_ROWS:
        g = r[m]["Test"]["by_same_sentence"]
        group_rows.append([name, inputs, f(g["same_sentence"]["f1"]), f(g["different_sentences"]["f1"])])
    structural, both = r["logreg_structural"]["Test"], r["logreg_both"]["Test"]
    return "\n\n".join([
        "## Q1 · 有没有关系",
        f"Test 共 {both['n']:,} 个候选对。指标针对有关系类。阈值在 Dev 上按 F1 最大选定，Test 沿用同一阈值。"
        "同句规则没有连续分数，AP 用 0/1 预测计算，仅作参照。",
        table(["模型", "输入", "Precision", "Recall", "F1", "AP", "Dev F1"], main_rows),
        f"**文本增益**（逻辑回归，结构加文本减仅结构）：F1 {both['f1'] - structural['f1']:+.3f}，"
        f"AP {both['average_precision'] - structural['average_precision']:+.3f}。",
        f"**按是否同句分组的 Test F1**（同句 n = {groups['same_sentence']['n']:,}，"
        f"不同句 n = {groups['different_sentences']['n']:,}）",
        table(["模型", "输入", "同句", "不同句"], group_rows),
    ])


def q2_section():
    r = {model: load("q2", model) for model, _, _ in Q2_ROWS}
    test = r["logreg_both"]["Test"]
    counts = {label: test["per_class"][label]["n"] for label in RELATION_TYPES}
    main_rows = [
        [name, inputs, f(r[m]["Test"]["macro_f1_8_classes"]), f(r[m]["Dev"]["macro_f1_8_classes"]),
         f(r[m]["Dev"]["macro_f1_present_classes"])]
        for m, name, inputs in Q2_ROWS
    ]
    class_rows = [
        [name, inputs, *(f(r[m]["Test"]["per_class"][label]["f1"]) for label in RELATION_TYPES)]
        for m, name, inputs in Q2_ROWS
    ]
    matrix = test["confusion_matrix"]["rows_true_columns_predicted"]
    matrix_rows = [[f"{SHORT[label]} ({counts[label]})", *row] for label, row in zip(RELATION_TYPES, matrix)]
    structural = r["logreg_structural"]["Test"]["macro_f1_8_classes"]

    predictions = pd.read_csv(OUTPUT_DIR / "predictions" / "q2_logreg_both_Test.csv", dtype=str)
    rare = predictions[predictions["relation_type"].isin(["Drug_Interaction", "Conversion"])]
    rare_rows = [[p.pmid, p.pair_type, p.entity1, p.entity2, p.relation_type, p.predicted] for p in rare.itertuples()]

    return "\n\n".join([
        "## Q2 · 是哪种关系",
        f"Test 共 {test['n']:,} 个已标注实体对。Dev 没有 Drug_Interaction 和 Conversion，"
        "所以 Dev 另列只平均出现过的六类的 macro-F1。",
        table(["模型", "输入", "Test 八类 macro-F1", "Dev 八类 macro-F1", "Dev 六类 macro-F1"], main_rows),
        f"**文本增益**（逻辑回归，结构加文本减仅结构）：Test 八类 macro-F1 "
        f"{test['macro_f1_8_classes'] - structural:+.3f}。",
        "**各类 Test F1**（表头括号内为 Test 样本数）",
        table(["模型", "输入", *(f"{SHORT[label]} ({counts[label]})" for label in RELATION_TYPES)], class_rows),
        "**混淆矩阵**（逻辑回归，结构加文本，Test。行为真实类别，列为预测类别）",
        table(["真实 \\ 预测", *(SHORT[label] for label in RELATION_TYPES)], matrix_rows, text_columns=1),
        "**Drug_Interaction 和 Conversion 的逐条预测**（逻辑回归，结构加文本）",
        table(["PMID", "实体对类型", "实体 1", "实体 2", "真实", "预测"], rare_rows, text_columns=6),
    ])


def main():
    parts = [
        "# Baseline 结果",
        "由 `python baseline/make_results_table.py` 从 `baseline/outputs/results/` 生成。"
        "输入为 `sota/prepare_pairs.py` 生成的共享表，结构特征为 `same_sentence`、`min_token_gap`、两个实体的提及次数和 `pair_type`，"
        "文本为 `context_local` 的 TF-IDF 词和双词短语。模型参数按 [Baseline 开工指南](README.md)，没有另行调参。"
        "各模型在 Train 拟合，Dev 只用于选 Q1 阈值，Test 各跑一次。",
        q1_section(),
        q2_section(),
        "## 复跑",
        "```\n"
        "python sota/prepare_pairs.py\n"
        "python baseline/same_sentence_rule.py --test\n"
        "python baseline/q1_models.py --model logreg --features structural --test   # 另有 text、both\n"
        "python baseline/q1_models.py --model svm --features text --test\n"
        "python baseline/q1_models.py --model nb --features text --test\n"
        "python baseline/q2_majority.py --test\n"
        "python baseline/q2_models.py --model logreg --features structural --test   # 另有 text、both\n"
        "python baseline/q2_models.py --model nb --features text --test\n"
        "python baseline/make_results_table.py\n"
        "```",
        "逐样本预测在 `baseline/outputs/predictions/`，每行带共享样本键（`split`、`pmid`、`pair_type` 和排序后的两个实体 ID）。"
        "Q1 文件含有关系分数 `score`，逻辑回归和朴素贝叶斯为概率，SVM 为 `decision_function`。依赖版本见 `baseline/requirements.txt`。",
    ]
    TARGET.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"Saved {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
