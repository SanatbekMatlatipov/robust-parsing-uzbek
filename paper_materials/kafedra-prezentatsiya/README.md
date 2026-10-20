# Department seminar talk: building the PDFs

LaTeX sources for the 15-minute department seminar talk (in Uzbek)
«O‘zbek tili gaplarini neyron tarmoqlar asosida sintaktik tahlil qilishning matematik modeli»,
specialty 05.01.07 (Mathematical modelling, numerical methods and software).

| Source file | Output | What it is |
|---|---|---|
| `Matlatipov_seminar_taqdimot.tex` | `Matlatipov_seminar_taqdimot.pdf` | Slides (Beamer, Madrid/dolphin theme as in the HonsAnSkills template, 16:9, 19 slides) |
| `Matlatipov_seminar_nutq_matni.tex` | `Matlatipov_seminar_nutq_matni.pdf` | Speech notes (A4, 14 pt): timed text per slide, key numbers, pre-seminar checklist, likely questions |
| `nuu-logo.png` | | University emblem on the title slide |

All charts and diagrams are drawn in TikZ/pgfplots from numbers embedded in the `.tex` file, so no other files are needed.

## Build

Requires a full TeX distribution (TeX Live 2025 / MacTeX / MiKTeX) with `beamer`, `pgfplots`,
`tikz-dependency`, `fontawesome5` and `extsizes`.

```bash
latexmk -pdf -outdir=build Matlatipov_seminar_taqdimot.tex
latexmk -pdf -outdir=build Matlatipov_seminar_nutq_matni.tex
cp build/*.pdf . && rm -rf build
```

Without `latexmk`, run `pdflatex` twice per file (the orange «Ilmiy yangilik N» footer tag is placed
using the previous pass).

## Where the numbers come from

| Slide | Numbers | Source in this repository |
|---|---|---|
| 2, 10, 11, 13 | corpus sizes, non-projectivity, arc lengths, head direction, relation counts | computed from `data/udbase/*/*.conllu` |
| 8 | BPE splits (kitob · larimiz · dagi …) | TahrirchiBERT tokenizer files (`vocab.json`, `merges.txt`) |
| 12, 14 | validation scores at the best checkpoint, learning curves | `saved_models/{pos,depparse}/*_training_log.csv` |
| 14 | paired bootstrap (B = 10 000) | `saved_models/depparse/*.dev.pred.conllu` vs. the dev files |
| 15 | test UAS/LAS and perturbation deltas (E2.1) | `paper_materials/review_logs/perturbation_results.json` |
| 15 | spaCy S1.1 test scores | `results/spacy_s1.1_test.json` |
| 9, 12 | hyperparameters | W&B run configs in `wandb/*/files/config.yaml` |

## Check before the seminar

- The scientific advisor on the title slide (`\ilmiyrahbar` in the slides preamble) and the
  publication statuses on slide 18 were inferred from the repository and must be confirmed.
- The "test" tables in the papers are dev-set scores, and the merged-data parsers were trained
  with gold POS/morphology. The slides label every number accordingly. The full list of issues is on
  pages 3–4 of the speech notes («Seminardan oldin tekshiring»).
