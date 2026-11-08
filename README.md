# Uzbek Dependency Parsing

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/Sanatbek/uzudt)
[![License](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

Code, data and results for neural morphosyntactic tagging and dependency parsing of Uzbek on the two Universal Dependencies (UD) treebanks, and for the analysis of dependency trees as spanning arborescences: exact laws of random trees, exact single-root decoding, matrix-tree normalization and temperature calibration of the tree distribution.

## Repository structure

```
robust-parsing-uzbek/
├── stanza/            # Stanza with BERT contextual embeddings for the tagger and the parser
├── spacy_uzbek/       # spaCy language module for Uzbek
├── scripts/           # evaluation, inference and treebank statistics
├── analysis/
│   ├── scripts/       # tree statistics, decoders, matrix-tree calibration, significance tests
│   └── results/       # computed results: JSON, CSV, score matrices and parsed test sets
├── config/            # environment paths and the XPOS vocabulary
├── data/              # UD treebanks and the prepared training files
├── saved_models/      # training curves (checkpoints are on Hugging Face)
├── wandb/, logs/      # training logs
└── results/           # spaCy evaluation results
```

## Data

| Treebank | Sentences | Words | Genre | UD release |
|----------|-----------|-------|-------|------------|
| UD_Uzbek-UzUDT | 684 | 7,582 | literary and educational | 2.17 |
| UD_Uzbek-UT | 500 | 5,930 | news and fiction | 2.15 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e stanza/             # the modified Stanza in this repository
```

FastText vectors for the static-embedding models (one-time):

```bash
curl -L -o wordvec/uz/fasttext/cc.uz.300.vec.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uz.300.vec.gz
gunzip -k wordvec/uz/fasttext/cc.uz.300.vec.gz
python -c "from stanza.models.common.pretrain import Pretrain; Pretrain('wordvec/uz/pretrain/fasttext_cc_uz_300.pt', 'wordvec/uz/fasttext/cc.uz.300.vec').vocab"
```

## Trained models

Seven parsers, each paired with a tagger trained in the same configuration. The checkpoints are at [huggingface.co/Sanatbek/uzudt](https://huggingface.co/Sanatbek/uzudt); place them in `saved_models/pos/` and `saved_models/depparse/`.

| Model | Word vectors | Subword aggregation | Training data |
|-------|--------------|---------------------|---------------|
| E1.1 | FastText | — | UzUDT |
| E2.1 | TahrirchiBERT | last piece | UzUDT |
| E3.1 | TahrirchiBERT | mean | UzUDT |
| E5.1 | BERTbek | mean | UzUDT |
| E1.2 | FastText | — | UzUDT + UT |
| E2.2 | TahrirchiBERT | last piece | UzUDT + UT |
| E3.2 | TahrirchiBERT | mean | UzUDT + UT |

Encoders: `tahrirchi/tahrirchi-bert-base` and `elmurod1202/bertbek-news-big-cased`.

### Training

Example for E2.2 (tagger, then parser):

```bash
python -m stanza.models.tagger --mode train --lang uz --shorthand uz_combined \
  --train_file data/pos/merged/uz_combined.train.in.conllu \
  --eval_file data/pos/merged/uz_combined.dev.in.conllu \
  --bert_model tahrirchi/tahrirchi-bert-base --no_pretrain \
  --save_dir saved_models/pos --save_name uz_combined_E2.2_tagger.pt

python -m stanza.models.parser --mode train --lang uz --shorthand uz_combined \
  --train_file data/depparse/merged/uz_combined.train.in.conllu \
  --eval_file data/depparse/merged/uz_combined.dev.in.conllu \
  --bert_model tahrirchi/tahrirchi-bert-base --no_pretrain \
  --save_dir saved_models/depparse --save_name uz_combined_E2.2_parser.pt
```

Variants: add `--bert_pooling mean` for mean aggregation; replace `--bert_model ... --no_pretrain` by `--wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec` for FastText; use `data/{pos,depparse}/uz_uzudt.*` with `--shorthand uz_uzudt` for UzUDT only. Add `--wandb` to log the run to Weights & Biases.

### Evaluation

```bash
python scripts/eval.py GOLD.conllu SYSTEM.conllu            # UD shared-task metrics (UAS, LAS, ...)
python scripts/eval_pos.py --gold GOLD.conllu --system SYSTEM.conllu
```

## Analysis

The scripts in `analysis/scripts/` write to `analysis/results/`. Run them from that folder with the modified Stanza on the path:

```bash
cd analysis/scripts
export PYTHONPATH=../../stanza
python graph_stats.py
```

| Script | Computes |
|--------|----------|
| `unified_testset.py` | test sets without exact or near duplicates of training and development sentences |
| `run_inference.py`, `labels_inference.py` | arc-score matrices, decoded trees and label distributions of all models |
| `decoders.py`, `test_decoders.py` | greedy, single-root Chu-Liu/Edmonds, Eisner and minimum-Bayes-risk decoders; matrix-tree marginals and entropy; checks against exhaustive search |
| `summarize.py`, `las_ci.py`, `bootstrap.py` | accuracy, bootstrap intervals and paired tests with Holm's correction |
| `graph_stats.py`, `exact_baselines.py` | tree statistics and exact laws of random trees, verified by enumeration |
| `kernel_clic.py` | arc-length laws compared by the composite likelihood information criterion |
| `conditional_crossings.py` | crossings of random trees conditioned on the total arc length |
| `calibration.py`, `normalised_Z.py` | probability of a well-formed tree, reliability and tree entropy |
| `calibrate.py`, `calibrate_ext.py` | temperature calibration by bisection and by a safeguarded Newton method; bootstrap, isotonic and log-linear comparisons |
| `numerics_checks.py`, `extended_analysis.py` | conditioning, root reweighting, label calibration, marginal-likelihood fit, sentence-level pooled bootstrap |
| `ensemble.py` | mixtures of the calibrated parsers |
| `timing.py`, `exhaustive_timing.py` | decoding time and exhaustive search |
| `tokenizer_stats.py`, `training_curves.py` | subword segmentation of inflected words and training curves |

`run_inference.py` and `labels_inference.py` parse with the checkpoints; `calibrate.py`, `calibrate_ext.py`, `ensemble.py` and `numerics_checks.py` import Stanza and PyTorch; the other scripts read `analysis/results/` and need only NumPy and SciPy.

## spaCy module

`spacy_uzbek/` adds Uzbek to spaCy, so that a tagger, morphologizer and parser can be trained on the same treebanks.

```bash
pip install spacy spacy-transformers
pip install -e spacy_uzbek/
python spacy_uzbek/convert_conllu.py --convert-all
python -m spacy train spacy_uzbek/configs/config_transformer.cfg --output saved_models/spacy/transformer_combined \
  --paths.train spacy_uzbek/data/uz_combined.train.spacy --paths.dev spacy_uzbek/data/uz_combined.dev.spacy --gpu-id 0
python -m spacy evaluate saved_models/spacy/transformer_combined/model-best spacy_uzbek/data/uz_combined.test.spacy
```

Configurations in `spacy_uzbek/configs/`: `config_transformer.cfg` (TahrirchiBERT), `config_fasttext.cfg` (FastText vectors) and `config_static.cfg` (hash embeddings).

## Contact

Sanatbek Matlatipov, National University of Uzbekistan named after Mirzo Ulugbek, s.matlatipov@nuu.uz

## License

CC BY-SA 4.0, see [LICENSE](LICENSE).
