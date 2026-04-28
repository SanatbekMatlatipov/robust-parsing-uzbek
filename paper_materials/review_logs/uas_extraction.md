# UAS Extraction — Provenance for Manuscript Revision

All values are taken **verbatim** from existing log/JSON files in this
repository. No new training or full-test evaluation was run.

## Stanza experiments (test set; values consistent with `README.md` §Results)

| Run  | Embeddings           | Data       | UPOS  | XPOS  | UFeats | UAS   | LAS   |
|------|----------------------|------------|-------|-------|--------|-------|-------|
| E1.1 | FastText             | UzUDT      | 79.19 | 79.81 | 66.61  | 69.57 | 51.24 |
| E1.2 | FastText             | UzUDT+UT   | 80.26 | 83.20 | 66.98  | 72.27 | 62.40 |
| E2.1 | TahrirchiBERT (last) | UzUDT      | 82.45 | 80.90 | 65.37  | 72.05 | 54.19 |
| E2.2 | TahrirchiBERT (last) | UzUDT+UT   | 85.08 | 84.72 | 71.09  | 72.39 | 63.81 |
| E3.1 | TahrirchiBERT (mean) | UzUDT      | 82.76 | 81.37 | 65.22  | 69.10 | 51.55 |

Sources:
- `saved_models/depparse/uz_uzudt_E1.1_parser_summary.json` — `final_uas`
- `saved_models/depparse/uz_combined_E1.2_parser_summary.json` — `final_uas`
- `saved_models/depparse/uz_uzudt_E2.1_parser_summary.json` — `final_uas`
- `saved_models/depparse/uz_combined_E2.2_parser_summary.json` — best-checkpoint test eval (RESEARCH_LOG §6.5, table on line 511)
- `saved_models/depparse/uz_uzudt_E3.1_parser_summary.json` — `final_uas`

## spaCy experiments (test set)

| Run  | Data     | UPOS  | XPOS  | UFeats | UAS   | LAS   |
|------|----------|-------|-------|--------|-------|-------|
| S1.1 | UzUDT    | 86.50 | 86.72 | 50.55  | 67.72 | 45.35 |
| S1.2 | UzUDT+UT | 89.18 | 88.24 | 65.48  | 66.81 | 47.11 |

Sources:
- `results/spacy_s1.1_test.json` — fields `pos_acc`, `tag_acc`, `morph_acc`, `dep_uas`, `dep_las`
- `results/spacy_s1.2_test.json` — same fields

## Cross-treebank augmentation deltas (Merged − UzUDT)

| Metric | Stanza Δ | spaCy Δ |
|--------|---------:|--------:|
| UPOS   |    +2.63 |   +2.68 |
| XPOS   |    +3.82 |   +1.52 |
| UFeats |    +5.72 |  +14.93 |
| UAS    |    +0.34 |   −0.91 |
| LAS    |    +9.62 |   +1.76 |

These were already present in the manuscript (`tab:augmentation`).
