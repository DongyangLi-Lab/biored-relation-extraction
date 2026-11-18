# biored-relation-extraction

INLS 613 course project on extracting biomedical relations from BioRED abstracts.

- **Q1 · Relation detection** — Does BioRED annotate a relation between this pair of entities?
- **Q2 · Relation classification** — For an annotated pair, which of the eight relation types applies?

Both tracks use the same candidate pairs, official splits, and evaluation metrics. The baseline also compares structural features with and without textual context to measure what the text adds.

| Path | Contents |
| --- | --- |
| [Experiment design](docs/experiment-design.md) | Task scope, shared input and output, and evaluation |
| [Baseline](baseline/README.md) | Rules, logistic regression, linear SVM, naive Bayes, and feature recipes |
| [SOTA track](sota/README.md) | BioBERT pre-trained model plan for both tasks |
| [Data](data/README.md) | Candidate and relation-label tables and the original BioRED download |

From the repository root, run `python data/download_biored.py` to fetch the three original BioC JSON splits from NCBI. The two project CSVs are already in the repository.

To contribute the baseline, fork this repository, create a `baseline` branch in your fork, and open a pull request targeting `master`. Start with the rules and structural logistic regression while the shared text input is being prepared.
