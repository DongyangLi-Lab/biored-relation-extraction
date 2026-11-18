# BioRED project data

The two CSVs are project-derived tables containing entity identifiers, split assignments, and relation annotations. The original annotated abstracts are available from [NCBI BioRED](https://github.com/ncbi/BioRED).

| File | Rows | Use |
| --- | ---: | --- |
| [candidate_pairs.csv](candidate_pairs.csv) | 35,136 | Q1 samples and binary annotation labels |
| [relation_labels.csv](relation_labels.csv) | 6,498 | Q2 samples and eight-way relation labels |

Join the tables by split, PMID, entity-pair type, and the two sorted entity IDs. Read IDs as strings. See the [experiment design](../docs/experiment-design.md) for split counts and modeling inputs.

Run `python data/download_biored.py` from the repository root. It downloads the original NCBI archive, verifies its SHA-256, and saves `Train.BioC.JSON`, `Dev.BioC.JSON`, and `Test.BioC.JSON` under `data/raw/BioRED/`. The original splits contain 400, 100, and 100 abstracts respectively.

Source citation: Luo et al. (2022), *BioRED: a rich biomedical relation extraction dataset*. [Paper and dataset description](https://doi.org/10.1093/bib/bbac282)
