# Summary of LaTeX Patches Applied to `main_paper.tex`

| # | Section | Change | Reviewer point |
|---|---------|--------|----------------|
| 1 | Preamble | Added `\usepackage{tikz}` and `\usepackage{tikz-dependency}` | W1 |
| 2 | §4.1 Stanza pipeline (Subword Fusion paragraph) | Trimmed inline example; pointer to new §4.4 | W2 |
| 3 | **NEW §4.4 Subword Alignment for Agglutinative Morphology** + new Table III (`tab:subword_examples`) | Three worked WordPiece decompositions, typological argument, quantitative ablation justification, explicit limitation | W2 |
| 4 | §5 Results (label `sec:results` added) | Anchor for cross-reference from §4.4 | W2 |
| 5 | Table IV `tab:stanza_results` | **UAS column added** (col count 7→8). Caption now records the source JSON files. | W3 |
| 6 | Table V `tab:spacy_results` | **UAS column added** (col count 6→7). Caption records `dep_uas` source field. | W3 |
| 7 | **NEW §5.5 Qualitative Parsing Examples** with two tikz-dependency figures `fig:parse_success` (s658, 100 % UAS/LAS) and `fig:parse_failure` (s653, LVC error) | W1 |
| 8 | §7.4 Limitations → renamed **Limitations and Threats to Validity** (label `sec:limitations`) and substantially expanded with single-run caveat, bootstrap CI, default-hyperparameter rationale, framework scope (UDPipe 2.0 / Trankit / DiaParser / MaChAmp), subword granularity, annotation heterogeneity | W4, W5 |
| 9 | §8 Conclusion / Future work — rewritten last paragraph to enumerate the five reviewer-suggested directions (LLM augmentation, confusion matrix, sentence-length analysis, profiling, ensembling) | reviewer "ideas" |
| 10 | Bibliography | Added `straka2018udpipe` and `whitehouse2024llm` | W4, future work |

All numerical values inserted into tables come from existing artefacts in
the repository:

- `saved_models/depparse/*_summary.json` (`final_uas`, `final_las`,
  `best_dev_score`)
- `results/spacy_s1.{1,2}_test.json` (`dep_uas`, `dep_las`, `pos_acc`,
  `tag_acc`, `morph_acc`)

No training or full-test inference was performed during this revision.
