# BioRED project data

The two CSVs are project-derived tables containing entity identifiers, split assignments, and relation annotations. The original annotated abstracts are available from [NCBI BioRED](https://github.com/ncbi/BioRED).

| File | Rows | Use |
| --- | ---: | --- |
| [candidate_pairs.csv](candidate_pairs.csv) | 35,136 | Q1 candidate pairs, binary annotation labels, and the shared `same_sentence` feature |
| [relation_labels.csv](relation_labels.csv) | 6,498 | Q2 samples and eight-way relation labels |

Join the tables by split, PMID, entity-pair type, and the two sorted entity IDs. Read IDs as strings. See the [experiment design](../docs/experiment-design.md) for split counts and modeling inputs.

`same_sentence` equals 1 if the two entities have at least one pair of mentions in the same sentence and 0 otherwise. It is available for every candidate in every split. Use it for the same-sentence baseline and as a structural feature. The `same_sentence_mention` field in the relation label table is only available for annotated pairs.

Run `python data/download_biored.py` from the repository root. It downloads the original NCBI archive, verifies its SHA-256, and saves `Train.BioC.JSON`, `Dev.BioC.JSON`, and `Test.BioC.JSON` under `data/raw/BioRED/`. The original splits contain 400, 100, and 100 abstracts respectively.

For the shared local text and other structural features, run `python sota/prepare_pairs.py` from the repo root after downloading BioRED. The [preprocessing script](../sota/prepare_pairs.py) generates `data/processed/Train.csv`, `Dev.csv`, and `Test.csv`. Use `context_local` for the local text input and `same_sentence` for the shared sentence feature. Both scripts use the Python standard library.

Source citation: Luo et al. (2022), *BioRED: a rich biomedical relation extraction dataset*. [Paper and dataset description](https://doi.org/10.1093/bib/bbac282)
