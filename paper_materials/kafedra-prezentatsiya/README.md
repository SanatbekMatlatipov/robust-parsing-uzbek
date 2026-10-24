# Department seminar talk: building the PDFs

LaTeX sources for the 15-minute department seminar talk (in Uzbek)
«O‘zbek tili gaplarini neyron tarmoqlar asosida sintaktik tahlil qilishning matematik modeli»,
specialty 05.01.07 (Mathematical modelling, numerical methods and software).

This is the second revision, made after the first seminar. It uses graph-theory terms («o‘zak» instead of
«hokim»), opens with a corpus record, its tree and why trees matter for meaning, and states exactly which model
was changed. It also separates the two optimisation problems (decoding vs. training), adds a measured
n^(n-1) vs O(n^2) decoding-time experiment, and spells out the Adam training work and results.

| Source file | Output | What it is |
|---|---|---|
| `Matlatipov_seminar_taqdimot.tex` | `Matlatipov_seminar_taqdimot.pdf` | Slides (Beamer, Madrid/dolphin theme as in the HonsAnSkills template, 16:9, 22 slides + 1 backup slide) |
| `Matlatipov_seminar_nutq_matni.tex` | `Matlatipov_seminar_nutq_matni.pdf` | Speech notes (A4, 14 pt): feedback-to-slide map, timed text per slide (≈1,270 words, 15:00), key numbers, pre-seminar checklist, likely questions |
| `mst_vaqt_tajribasi.py` | `mst_vaqt_natijalari.json` | Decoding experiment: tree counts, CLE vs. exhaustive search (correctness and time) |
| `nuu-logo.png` | | University emblem on the title slide |

All charts and diagrams are drawn in TikZ/pgfplots from numbers embedded in the `.tex` file, so no other files are
needed to build the PDFs.

## Build

Requires a full TeX distribution (TeX Live 2025 / MacTeX / MiKTeX) with `beamer`, `pgfplots`,
`appendixnumberbeamer`, `fontawesome5` and `extsizes`.

```bash
latexmk -pdf -outdir=build Matlatipov_seminar_taqdimot.tex
latexmk -pdf -outdir=build Matlatipov_seminar_nutq_matni.tex
cp build/*.pdf . && rm -rf build
```

Without `latexmk`, run `pdflatex` twice per file (the orange «Ilmiy yangilik N» footer tag is placed
using the previous pass).

The decoding experiment needs only NumPy; it imports Stanza's own `chuliu_edmonds.py` from this repository
(about one minute on an Apple M2 Max):

```bash
python paper_materials/kafedra-prezentatsiya/mst_vaqt_tajribasi.py   # run from the repository root
```

## Where the numbers come from

| Slide | Numbers | Source in this repository |
|---|---|---|
| 2 | corpus record and its tree (sentence s658) | `data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-dev.conllu` |
| 4, 6, 11, 13, 16 | corpus sizes, non-projectivity, arc lengths, head direction, relation counts | computed from `data/udbase/*/*.conllu` |
| 6, 12–14 | tree counts n^(n-1), CLE = exhaustive optimum (1,560/1,560), timings, growth exponent | `mst_vaqt_tajribasi.py` → `mst_vaqt_natijalari.json` |
| 9, 12, 15 | trainable parameters: 24,513,613 (FastText parser) → 27,563,156 (BERT channel); TahrirchiBERT 109,486,848 (frozen) | `saved_models/depparse/uz_uzudt_nocharlm_parser.pt` (input 500) plus the input-layer change 500 → 1143; TahrirchiBERT `config.json` |
| 9, 10 | what was changed in Stanza | git commits `4ab3dbe`, `599afd3`, `bf9cfce` (`depparse/model.py`, `trainer.py`, `data.py`, `common/bert_embedding.py`, `pos/model.py`, `parser.py`, `tagger.py`, `constant.py`) |
| 10 | BPE splits (kitob · larimiz · dagi …) | TahrirchiBERT tokenizer files (`vocab.json`, `merges.txt`) |
| 12, 15, 17 | validation scores at the best checkpoint, E2.1 learning curve (loss and LAS per 100 steps) | `saved_models/{pos,depparse}/*_training_log.csv` |
| 17 | paired bootstrap (B = 10 000) | `saved_models/depparse/*.dev.pred.conllu` vs. the dev files |
| 18 | test UAS/LAS and perturbation deltas (E2.1) | `paper_materials/review_logs/perturbation_results.json` |
| 18 | spaCy S1.1 test scores | `results/spacy_s1.1_test.json` |
| 15, backup | hyperparameters | W&B run configs in `wandb/*/files/config.yaml` |

## Check before the seminar

- The scientific advisor on the title slide (`\ilmiyrahbar` in the slides preamble) and the
  publication statuses on slide 21 were inferred from the repository and must be confirmed.
- Chu–Liu–Edmonds and Adam are named on the slides, without bibliographic citations. The slides claim the
  work that was done in this project: formulating Uzbek parsing as a graph problem, modifying the Stanza
  parser, applying CLE in it, and verifying and timing it. They do not claim the algorithms themselves.
- «Ilk bor» (first) is used only where it can be defended. UD_Uzbek-UT's authors call theirs the first Uzbek UD
  treebank, so UzUDT is described as the first for literary and educational texts and as the largest.
- The "test" tables in the papers are dev-set scores, and the merged-data parsers were trained
  with gold POS/morphology. The slides label every number accordingly. The full list of issues is in the speech
  notes («Seminardan oldin tekshiring»).
