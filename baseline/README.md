# Baseline 开工指南

用 Python 和 scikit-learn 跑规则、逻辑回归、线性 SVM 和多项朴素贝叶斯。两项任务都使用 [共享数据](../data/README.md) 和 [共同实验设计](../docs/experiment-design.md)。

| 模型 | Q1 有没有关系 | Q2 是哪种关系 |
| --- | --- | --- |
| 简单参照 | 同句共现规则。只要两个实体至少有一对提及出现在同一句，就预测有关系。 | 训练集中出现最多的关系类型，作为多数类基线。 |
| 逻辑回归 | 跑仅结构、仅文本、结构加文本三组。 | 用多项逻辑回归跑相同三组输入，每个模型直接预测八类之一。 |
| 线性 SVM | 用与逻辑回归相同的仅文本特征，比较两种分类器。 | — |
| 多项朴素贝叶斯 | 复用仅文本特征。 | 复用仅文本特征，作为快速的八分类对照。 |

逻辑回归可以从 `LogisticRegression(solver="lbfgs", C=1, max_iter=1000)` 开始。线性 SVM 用 `LinearSVC(C=1)`，Q1 用 `decision_function` 的分数计算 average precision，并在 Dev 选阈值。多项朴素贝叶斯用 `MultinomialNB(alpha=1.0)`，输入限于非负的 TF-IDF。同一任务内，各模型复用词表、数据划分和评估代码。 [逻辑回归](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) · [线性 SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) · [多项朴素贝叶斯](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)

| 输入 | 怎么做 |
| --- | --- |
| **仅结构** | 从共享预处理表读取 `min_token_gap`、`entity1_mentions`、`entity2_mentions`、`same_sentence` 和 `pair_type`。词距和次数可用 `log1p` 再标准化，`pair_type` 用 one-hot。 |
| **仅文本** | 读取共享预处理表的 `context_local`，用 `TfidfVectorizer(ngram_range=(1, 2), min_df=2)` 提取词和双词短语。保留否定词。 |
| **结构加文本** | 将上面两个特征矩阵横向拼接。可以用 `ColumnTransformer` 和 `Pipeline`，保留 TF-IDF 的稀疏表示。 |

同句共现规则直接用候选表的 `same_sentence` 作预测。按 `split` 取 Train、Dev 和 Test。生成完整特征表的命令见共享数据说明。实体 ID 与 PMID 用来对齐样本。词表、标准化和模型在 Train 拟合，Dev 用来选择参数与 Q1 阈值，Test 留作最终评估。 [候选表](../data/candidate_pairs.csv) · [共享数据说明](../data/README.md) · [TF-IDF 参数说明](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

交付时带上可复跑的训练命令和逐样本预测，记录特征版本和参数。Q1 报告有关系类的 precision、recall、F1 和 average precision，逐样本预测同时给出有关系分数。Q2 报告八类 macro-F1、各类 F1、样本数和混淆矩阵。用逻辑回归的「结构加文本」和「仅结构」衡量文本增益，再用各模型的「仅文本」结果比较同一文本输入下的表现。
