# SOTA track · BioBERT

项目主导负责这条线。首轮采用 [BioBERT](https://pubmed.ncbi.nlm.nih.gov/31501885/) 基础预训练权重，在本项目 Train 上分别微调 Q1 二分类和 Q2 八分类。报告用 **BioBERT** 标识该模型。

1. 用 [共享数据](../data/README.md) 提取最近一对目标提及所在的句子，保留文档顺序，用 `ENTITY1` 和 `ENTITY2` 标记目标。将同一列文本交给 Baseline。
2. 用一小批训练样本确认两个标记和标签能正确读入。随后在官方 Train 上训练两个任务，使用 Dev 选模型。
3. 按 [实验设计](../docs/experiment-design.md) 交付逐样本预测。Q1 同时保存有关系分数，Q2 按固定的八类标签顺序生成混淆矩阵。用相同文本输入与逻辑回归对照。
