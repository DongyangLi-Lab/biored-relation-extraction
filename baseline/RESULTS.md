# Baseline 结果

由 `python baseline/make_results_table.py` 从 `baseline/outputs/results/` 生成。输入为 `sota/prepare_pairs.py` 生成的共享表，结构特征为 `same_sentence`、`min_token_gap`、两个实体的提及次数和 `pair_type`，文本为 `context_local` 的 TF-IDF 词和双词短语。模型参数按 [Baseline 开工指南](README.md)，没有另行调参。各模型在 Train 拟合，Dev 只用于选 Q1 阈值，Test 各跑一次。

## Q1 · 有没有关系

Test 共 7,047 个候选对。指标针对有关系类。阈值在 Dev 上按 F1 最大选定，Test 沿用同一阈值。同句规则没有连续分数，AP 用 0/1 预测计算，仅作参照。

| 模型 | 输入 | Precision | Recall | F1 | AP | Dev F1 |
|---|---|---:|---:|---:|---:|---:|
| 同句规则 | — | 0.347 | 0.806 | 0.486 | 0.312 | 0.475 |
| 逻辑回归 | 仅结构 | 0.600 | 0.740 | 0.662 | 0.641 | 0.690 |
| 逻辑回归 | 仅文本 | 0.474 | 0.613 | 0.535 | 0.529 | 0.556 |
| 逻辑回归 | 结构加文本 | 0.641 | 0.713 | 0.675 | 0.676 | 0.701 |
| 线性 SVM | 仅文本 | 0.476 | 0.542 | 0.507 | 0.504 | 0.550 |
| 多项朴素贝叶斯 | 仅文本 | 0.228 | 0.652 | 0.338 | 0.237 | 0.358 |

**文本增益**（逻辑回归，结构加文本减仅结构）：F1 +0.013，AP +0.035。

**按是否同句分组的 Test F1**（同句 n = 2,699，不同句 n = 4,348）

| 模型 | 输入 | 同句 | 不同句 |
|---|---|---:|---:|
| 同句规则 | — | 0.516 | 0.000 |
| 逻辑回归 | 仅结构 | 0.712 | 0.294 |
| 逻辑回归 | 仅文本 | 0.620 | 0.112 |
| 逻辑回归 | 结构加文本 | 0.732 | 0.226 |
| 线性 SVM | 仅文本 | 0.590 | 0.070 |
| 多项朴素贝叶斯 | 仅文本 | 0.488 | 0.153 |

## Q2 · 是哪种关系

Test 共 1,163 个已标注实体对。Dev 没有 Drug_Interaction 和 Conversion，所以 Dev 另列只平均出现过的六类的 macro-F1。

| 模型 | 输入 | Test 八类 macro-F1 | Dev 八类 macro-F1 | Dev 六类 macro-F1 |
|---|---|---:|---:|---:|
| 多数类 | — | 0.088 | 0.081 | 0.108 |
| 逻辑回归 | 仅结构 | 0.236 | 0.221 | 0.295 |
| 逻辑回归 | 仅文本 | 0.166 | 0.163 | 0.217 |
| 逻辑回归 | 结构加文本 | 0.265 | 0.258 | 0.344 |
| 多项朴素贝叶斯 | 仅文本 | 0.112 | 0.111 | 0.147 |

**文本增益**（逻辑回归，结构加文本减仅结构）：Test 八类 macro-F1 +0.029。

**各类 Test F1**（表头括号内为 Test 样本数）

| 模型 | 输入 | Assoc (635) | Pos (325) | Neg (171) | Bind (9) | Cotreat (14) | Compar (6) | DrugInt (2) | Conver (1) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 多数类 | — | 0.706 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 逻辑回归 | 仅结构 | 0.779 | 0.438 | 0.308 | 0.000 | 0.364 | 0.000 | 0.000 | 0.000 |
| 逻辑回归 | 仅文本 | 0.754 | 0.381 | 0.196 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 逻辑回归 | 结构加文本 | 0.794 | 0.506 | 0.399 | 0.000 | 0.421 | 0.000 | 0.000 | 0.000 |
| 多项朴素贝叶斯 | 仅文本 | 0.726 | 0.124 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**混淆矩阵**（逻辑回归，结构加文本，Test。行为真实类别，列为预测类别）

| 真实 \ 预测 | Assoc | Pos | Neg | Bind | Cotreat | Compar | DrugInt | Conver |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Assoc (635) | 557 | 54 | 24 | 0 | 0 | 0 | 0 | 0 |
| Pos (325) | 144 | 148 | 32 | 0 | 1 | 0 | 0 | 0 |
| Neg (171) | 55 | 56 | 60 | 0 | 0 | 0 | 0 | 0 |
| Bind (9) | 8 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Cotreat (14) | 3 | 0 | 7 | 0 | 4 | 0 | 0 | 0 |
| Compar (6) | 0 | 1 | 5 | 0 | 0 | 0 | 0 | 0 |
| DrugInt (2) | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Conver (1) | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Drug_Interaction 和 Conversion 的逐条预测**（逻辑回归，结构加文本）

| PMID | 实体对类型 | 实体 1 | 实体 2 | 真实 | 预测 |
|---|---|---|---|---|---|
| 18503483 | Chemical+Chemical | D000068338 | D016559 | Conversion | Association |
| 15811908 | Chemical+Chemical | D000077582 | D063325 | Drug_Interaction | Positive_Correlation |
| 19531695 | Chemical+Chemical | D005424 | D017374 | Drug_Interaction | Negative_Correlation |

## 文本增益的稳定性

配对 bootstrap：按摘要对 Test 有放回重抽 2,000 次，每次两个逻辑回归模型使用同一批摘要，计算「结构加文本」减「仅结构」的差值（随机种子 0）。

| 指标 | 增益 | 95% 区间 | 增益大于 0 的比例 | 稳定提升 |
|---|---:|---:|---:|---:|
| Q1 F1 | +0.013 | -0.008 至 +0.032 | 89.3% | 否，区间含 0 |
| Q1 AP | +0.035 | +0.017 至 +0.051 | 100.0% | 是 |
| Q2 八类 macro-F1 | +0.029 | +0.010 至 +0.053 | 100.0% | 是 |

95% 区间全部在 0 以上，才认为文本带来稳定提升；区间含 0 时，提升与抽样波动无法区分。

## 复跑

```
python sota/prepare_pairs.py
python baseline/same_sentence_rule.py --test
python baseline/q1_models.py --model logreg --features structural --test   # 另有 text、both
python baseline/q1_models.py --model svm --features text --test
python baseline/q1_models.py --model nb --features text --test
python baseline/q2_majority.py --test
python baseline/q2_models.py --model logreg --features structural --test   # 另有 text、both
python baseline/q2_models.py --model nb --features text --test
python baseline/make_results_table.py
```

逐样本预测在 `baseline/outputs/predictions/`，每行带共享样本键（`split`、`pmid`、`pair_type` 和排序后的两个实体 ID）。Q1 文件含有关系分数 `score`，逻辑回归和朴素贝叶斯为概率，SVM 为 `decision_function`。依赖版本见 `baseline/requirements.txt`。
