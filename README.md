# Towards Robust Uzbek Neural Dependency Parsing

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/Sanatbek/uzudt)
[![License](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

This repository contains the official implementation, evaluation pipelines, and scripts for the paper *Towards Robust Uzbek Neural Dependency Parsing*.

We train and evaluate **Stanza-style neural pipelines** for Uzbek morphosyntactic tagging (UPOS/XPOS/UFeats) and UD dependency parsing (UAS/LAS), comparing a **static FastText baseline** against **TahrirchiBERT contextual embeddings** across two Uzbek UD treebanks.

---

## Repository Structure

```
robust-parsing-uzbek/
├── stanza/                         # Modified Stanza framework (BERT-enabled POS & depparse)
│   └── stanza/models/
│       ├── tagger.py               # POS tagger entry point (--bert_model supported)
│       ├── parser.py               # Dependency parser entry point (--bert_model supported)
│       ├── pos/                    # POS model, trainer, data, scorer
│       ├── depparse/               # Depparse model, trainer, data, scorer
│       └── common/
│           └── bert_embedding.py   # BERT extraction with pooling strategies
├── spacy_uzbek/                    # spaCy Uzbek language support (NEW)
│   ├── setup.py                    # Package installer (pip install -e spacy_uzbek/)
│   ├── lang/uz/                    # Custom Uzbek Language class for spaCy
│   │   ├── __init__.py             # Language class + registry entry point
│   │   ├── stop_words.py           # Uzbek stop words
│   │   └── tokenizer_exceptions.py # Abbreviation & MWT handling
│   ├── configs/
│   │   ├── config_transformer.cfg  # TahrirchiBERT transformer pipeline
│   │   ├── config_static.cfg       # Hash embeddings (no pretrained vectors)
│   │   └── config_fasttext.cfg     # FastText static vectors pipeline
│   ├── convert_conllu.py           # CoNLL-U → spaCy DocBin converter
│   ├── train.py                    # End-to-end training orchestrator
│   ├── evaluate.py                 # Evaluation & per-tag metrics
│   └── data/                       # Converted .spacy binary files
├── scripts/
│   ├── eval.py                     # CoNLL 2018 shared task evaluation
│   ├── eval_pos.py                 # POS accuracy evaluation
│   ├── eval_upos_by_tag.py         # Per-tag POS breakdown
│   ├── parse_test_pos_only.py      # POS-only inference script
│   └── parse_test_with_depparse.py # Full parsing inference script
├── config/
│   ├── config.sh                   # Environment paths for Stanza training
│   └── xpos_vocab_factory.py       # XPOS vocabulary factory (incl. Uzbek)
├── data/
│   ├── udbase/
│   │   ├── UD_Uzbek-UzUDT/         # UzUDT treebank (Matlatipov, 684 sents)
│   │   │   ├── uz_uzudt-ud-train.conllu  (451 sents)
│   │   │   ├── uz_uzudt-ud-dev.conllu    (45 sents)
│   │   │   └── uz_uzudt-ud-test.conllu   (188 sents)
│   │   └── UD_Uzbek-UT/            # UT treebank (Akhundjanova & Talamo, 500 sents)
│   │       ├── uz_ut-ud-train.conllu     (330 sents)
│   │       ├── uz_ut-ud-dev.conllu       (33 sents)
│   │       └── uz_ut-ud-test.conllu      (137 sents)
│   ├── pos/                        # Processed POS training data
│   └── depparse/                   # Processed depparse training data
├── wordvec/uz/                     # FastText cc.uz.300.vec (static embeddings)
├── saved_models/
│   ├── pos/                        # Trained POS tagger checkpoints
│   ├── depparse/                   # Trained parser checkpoints
│   └── spacy/                      # Trained spaCy pipeline models (NEW)
├── logs/                           # Training logs
├── RESEARCH_LOG.md                 # Detailed research log and findings
├── future_research_log.md          # Planned future experiments (E3-E7)
└── research-paper.tex              # LaTeX source of the paper
```

---

## Datasets

We evaluate on two Uzbek UD treebanks:

| Treebank | Author(s) | Sentences | Train / Dev / Test | Genre | UD Release |
|----------|-----------|-----------|-------------------|-------|------------|
| **UD_Uzbek-UzUDT** | Matlatipov, Sanatbek | 684 | 451 / 45 / 188 | Literature, Academic | v2.17 |
| **UD_Uzbek-UT** | Akhundjanova & Talamo | 500 | 330 / 33 / 137 | News, Fiction | v2.15 |

Both treebanks are split into train/dev/test using consistent ~66%/7%/27% proportions.

---

## Methodology

<p align="center">
  <img src="baseline_architecture.png" alt="Baseline Uzbek UD parsing framework architecture" width="85%">
  <br>
  <em><b>Figure 1:</b> Baseline Uzbek UD parsing framework.</em>
</p>

The pipeline has two stages:

1. **Contextual Embedding** — A monolingual Uzbek BERT encodes tokens into subword representations, which are then aggregated to UD word tokens via "super-token" fusion (mean pooling or last-subword selection).
2. **Joint Tagging & Parsing** — A BiLSTM + DeepBiaffine architecture predicts UPOS/XPOS/UFeats tags and dependency arcs+relations.

### Transformer Model

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| **TahrirchiBERT** | `tahrirchi/tahrirchi-bert-base` | Uzbek monolingual BERT, 768-dim hidden, broad domain |

---

## Setup

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -U pip
pip install -r requirements.txt
```

### 2. Install Stanza (local modified version)

The modified Stanza code is included in `stanza/`. Install it in editable (development) mode so changes are reflected immediately:

```bash
pip install -e stanza/
```

Verify it works:

```bash
python -m stanza.models.tagger --help
python -m stanza.models.parser --help
```

### 3. Pretrained Word Vectors

Download and prepare FastText Uzbek vectors (one-time setup):

```powershell
# Download (~469 MB compressed)
curl.exe -L -o wordvec/uz/fasttext/cc.uz.300.vec.gz `
  https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uz.300.vec.gz

# Decompress
python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('wordvec/uz/fasttext/cc.uz.300.vec.gz','rb'), open('wordvec/uz/fasttext/cc.uz.300.vec','wb'))"

# Convert to Stanza .pt format
python -c "
import os; os.makedirs('wordvec/uz/pretrain', exist_ok=True)
from stanza.models.common.pretrain import Pretrain
pt = Pretrain('wordvec/uz/pretrain/fasttext_cc_uz_300.pt', 'wordvec/uz/fasttext/cc.uz.300.vec')
_ = pt.vocab  # triggers load + save
print('Saved', 'wordvec/uz/pretrain/fasttext_cc_uz_300.pt')
"
```

### 4. Weights & Biases (optional but recommended)

We use [W&B](https://wandb.ai) to track loss, accuracy, and learning rate across all runs. Login once:

```bash
wandb login
```

Add `--wandb` to any training command to enable W&B logging. All runs are also logged locally as CSV + PNG plots regardless of W&B.

---

## Experiments

We run 2 experiment configurations comparing a static-embedding baseline against contextual BERT embeddings. Each configuration is trained on **two data settings**:

- **X.1** — UzUDT only (train 451 / dev 45 / test 188 sentences)
- **X.2** — UzUDT + UT merged (train 781 / dev 78 / test 325 sentences)

This gives **4 total runs** (2 configurations × 2 data settings).

**Workflow per run:** Train POS tagger â†’ Re-tag data â†’ Train dependency parser â†’ Evaluate on test set

### Experiment Matrix

| Exp | Data | Transformer | Fusion | Static Pretrain |
|-----|------|-------------|--------|-----------------|
| E1.1 | UzUDT | — | — | FastText |
| E1.2 | UzUDT+UT | — | — | FastText |
| E2.1 | UzUDT | TahrirchiBERT | last-subword | — |
| E2.2 | UzUDT+UT | TahrirchiBERT | last-subword | — |

### Data Files

| Setting | Train file | Dev file | Test file | Train | Dev | Test |
|---------|-----------|----------|-----------|-------|-----|------|
| UzUDT (`.1`) | `data/pos/uz_uzudt.train.in.conllu` | `data/pos/uz_uzudt.dev.in.conllu` | `data/pos/uz_uzudt.test.in.conllu` | 435 | 48 | 198 |
| UzUDT+UT (`.2`) | `data/pos/merged/uz_combined.train.in.conllu` | `data/pos/merged/uz_combined.dev.in.conllu` | `data/pos/merged/uz_combined.test.in.conllu` | 781 | 78 | 325 |

> **Note:** Depparse uses the same file names under `data/depparse/` and `data/depparse/merged/` respectively.

### Training Commands

#### E1.1: FastText-only baseline — UzUDT

```powershell
# Step 1: Train POS tagger
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/pos/uz_uzudt.train.in.conllu `
  --eval_file data/pos/uz_uzudt.dev.in.conllu `
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt `
  --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec `
  --save_dir saved_models/pos --save_name uz_uzudt_E1.1_tagger.pt `
  --wandb

# Step 2: Train dependency parser
python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/depparse/uz_uzudt.train.in.conllu `
  --eval_file data/depparse/uz_uzudt.dev.in.conllu `
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt `
  --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec `
  --save_dir saved_models/depparse --save_name uz_uzudt_E1.1_parser.pt `
  --wandb
```

#### E1.2: FastText-only baseline — UzUDT+UT merged

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/pos/merged/uz_combined.train.in.conllu `
  --eval_file data/pos/merged/uz_combined.dev.in.conllu `
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt `
  --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec `
  --save_dir saved_models/pos --save_name uz_combined_E1.2_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/depparse/merged/uz_combined.train.in.conllu `
  --eval_file data/depparse/merged/uz_combined.dev.in.conllu `
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt `
  --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec `
  --save_dir saved_models/depparse --save_name uz_combined_E1.2_parser.pt `
  --wandb
```

#### E2.1: TahrirchiBERT + last-subword — UzUDT

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/pos/uz_uzudt.train.in.conllu `
  --eval_file data/pos/uz_uzudt.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --no_pretrain `
  --save_dir saved_models/pos --save_name uz_uzudt_E2.1_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/depparse/uz_uzudt.train.in.conllu `
  --eval_file data/depparse/uz_uzudt.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --no_pretrain `
  --save_dir saved_models/depparse --save_name uz_uzudt_E2.1_parser.pt `
  --wandb
```

#### E2.2: TahrirchiBERT + last-subword — UzUDT+UT merged

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/pos/merged/uz_combined.train.in.conllu `
  --eval_file data/pos/merged/uz_combined.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --no_pretrain `
  --save_dir saved_models/pos --save_name uz_combined_E2.2_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/depparse/merged/uz_combined.train.in.conllu `
  --eval_file data/depparse/merged/uz_combined.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --no_pretrain `
  --save_dir saved_models/depparse --save_name uz_combined_E2.2_parser.pt `
  --wandb
```

#### E3.1: TahrirchiBERT + mean pooling — UzUDT

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/pos/uz_uzudt.train.in.conllu `
  --eval_file data/pos/uz_uzudt.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --bert_pooling mean `
  --no_pretrain `
  --save_dir saved_models/pos --save_name uz_uzudt_E3.1_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/depparse/uz_uzudt.train.in.conllu `
  --eval_file data/depparse/uz_uzudt.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --bert_pooling mean `
  --no_pretrain `
  --save_dir saved_models/depparse --save_name uz_uzudt_E3.1_parser.pt `
  --wandb
```

#### E3.2: TahrirchiBERT + mean pooling — UzUDT+UT merged

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/pos/merged/uz_combined.train.in.conllu `
  --eval_file data/pos/merged/uz_combined.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --bert_pooling mean `
  --no_pretrain `
  --save_dir saved_models/pos --save_name uz_combined_E3.2_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/depparse/merged/uz_combined.train.in.conllu `
  --eval_file data/depparse/merged/uz_combined.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --bert_pooling mean `
  --no_pretrain `
  --save_dir saved_models/depparse --save_name uz_combined_E3.2_parser.pt `
  --wandb
```

### Evaluation

```powershell
# POS accuracy — UzUDT test
python scripts/eval_pos.py `
  --gold data/pos/uz_uzudt.test.in.conllu `
  --system saved_models/pos/uz_uzudt_E1.1_tagger.pred.conllu

# POS accuracy — Combined test
python scripts/eval_pos.py `
  --gold data/pos/merged/uz_combined.test.in.conllu `
  --system saved_models/pos/uz_combined_E1.2_tagger.pred.conllu

# Full UD metrics (UAS, LAS, CLAS, MLAS, BLEX)
python scripts/eval.py `
  data/pos/uz_uzudt.test.in.conllu `
  saved_models/depparse/uz_uzudt_E1.1_parser.pred.conllu
```

---


## Training Outputs & Visualization

Every training run automatically produces:

| File | Description |
|------|-------------|
| `*_training_log.csv` | Step-level metrics (loss, dev scores, LR, wall time) |
| `*_summary.json` | Best scores, hyperparameters, total training time |
| `*_loss.png` | Training loss curve |
| `*_accuracy.png` | Dev accuracy curves (UPOS/XPOS/UFeats or UAS/LAS) |
| `*_lr.png` | Learning rate schedule |
| `*_overview.png` | Combined loss + primary metric (dual-axis) |

### Comparing experiments

After running multiple experiments, generate merged comparison plots:

```powershell
# POS tagger comparison
python scripts/compare_experiments.py --mode pos `
  --experiments saved_models/pos/uz_uzudt_E1_training_log.csv saved_models/pos/uz_uzudt_E2.1_training_log.csv `
  --labels E1.1-FastText E2.1-TahrirchiBERT `
  --output_dir results/comparison_plots

# Dependency parser comparison
python scripts/compare_experiments.py --mode depparse `
  --experiments saved_models/depparse/uz_uzudt_E1.1_training_log.csv saved_models/depparse/uz_uzudt_E2.1_training_log.csv `
  --labels E1.1-FastText E2.1-TahrirchiBERT `
  --output_dir results/comparison_plots

# Bar chart from JSON summaries
python scripts/compare_experiments.py --mode summary `
  --experiments saved_models/pos/uz_uzudt_E1_summary.json saved_models/pos/uz_uzudt_E2.1_summary.json `
  --labels E1.1 E2.1 `
  --output_dir results/comparison_plots
```

---

## Results

The table below summarizes the main experimental results; see `RESEARCH_LOG.md` for detailed analysis and experimental chronology.

| Exp | Data | Embeddings | Fusion | UPOS | XPOS | UFeats | UAS | LAS |
|-----|------|------------|--------|------|------|--------|-----|-----|
| E1.1 | UzUDT | FastText | N/A | 79.19 | 79.81 | 66.61 | 69.57 | 51.24 |
| E1.2 | UzUDT+UT | FastText | N/A | 80.26 | 83.20 | 66.98 | 72.27 | 62.40 |
| E2.1 | UzUDT | TahrirchiBERT | Last-sub | 82.45 | 80.90 | 65.37 | 72.05 | 54.19 |
| E2.2 | UzUDT+UT | TahrirchiBERT | Last-sub | **85.08** | 84.72 | **71.09** | **72.39** | **63.81** |
| E3.1 | UzUDT | TahrirchiBERT | Mean | 82.76 | 81.37 | 65.22 | 69.10 | 51.55 |
| E3.2 | UzUDT+UT | TahrirchiBERT | Mean | 84.02 | **87.07** | 70.39 | 70.74 | 60.05 |

### Key Findings

1. **TahrirchiBERT outperforms FastText on POS tagging.** The best BERT model (E2.2) achieves 85.08 UPOS vs 80.26 for the best FastText model (E1.2), a gain of **+4.82 points**.
2. **Merging treebanks consistently helps.** Every `.2` run outperforms its `.1` counterpart, with LAS gains of +11.16 (E1) and +9.62 (E2).
3. **Fusion strategy is task-dependent.** Last-subword is clearly better for parsing (+2.95 UAS, +2.64 LAS on UzUDT). Mean pooling is better for XPOS (E3.2 reaches 87.07, the highest across all runs).
4. **Best overall system: E2.2** — TahrirchiBERT + last-subword + UzUDT+UT merged data achieves the highest scores in UPOS, UFeats, UAS, and LAS.

---

## Key Ablations

The experiments are designed to answer two questions:

1. **Does TahrirchiBERT improve over static FastText embeddings?** Yes: +4.82 UPOS, +1.41 LAS on merged data.
2. **Does merging two Uzbek UD treebanks help?** Yes: the combined data consistently yields higher scores, especially LAS (+11.16 for E1, +9.62 for E2).
3. **Does fusion strategy matter?** It depends on the task: last-subword is better for parsing; mean pooling is competitive or better for some POS metrics (see `RESEARCH_LOG.md` §6.4 and §6.9).

> **Future work:** Additional experiments with BERTbek and BERT+FastText fusion are planned — see `future_research_log.md`.

---

## spaCy Pipeline for Uzbek

Since spaCy does not natively support Uzbek, this project includes a **custom spaCy language module** (`spacy_uzbek/`) that enables training spaCy-based POS taggers, morphologizers, and dependency parsers for Uzbek using the same UD treebank data.

### Why spaCy?

| Feature | Stanza (primary) | spaCy (new) |
|---------|-------------------|-------------|
| Architecture | BiLSTM + DeepBiaffine | Transition-based + Tok2Vec |
| Transformer | HuggingFace BERT (custom integration) | `spacy-transformers` (native) |
| Inference speed | Moderate | Fast (Cython optimized) |
| Packaging | Research-oriented | Production-ready (`spacy package`) |
| Language support | Uzbek via custom scripts | Uzbek via `spacy_uzbek/` entry point |

### Resources Required

The spaCy pipeline reuses the same UD treebank data already in this repository:

| Resource | Location | Notes |
|----------|----------|-------|
| UD_Uzbek-UzUDT | `data/udbase/UD_Uzbek-UzUDT/` | 684 sentences |
| UD_Uzbek-UT | `data/udbase/UD_Uzbek-UT/` | 500 sentences |
| Processed POS files | `data/pos/`, `data/pos/merged/` | CoNLL-U for both treebanks |
| FastText vectors | `wordvec/uz/fasttext/cc.uz.300.vec` | Optional, for static configs |
| TahrirchiBERT | `tahrirchi/tahrirchi-bert-base` | HuggingFace, for transformer config |
| spaCy ≥ 3.5 | `pip install spacy` | Core framework |
| spacy-transformers ≥ 1.2 | `pip install spacy-transformers` | For BERT-based training |

### Setup

#### 1. Install spaCy + Uzbek language support

```powershell
# Install spaCy and transformer support
pip install spacy spacy-transformers spacy-loggers

# Install the custom Uzbek language package (editable mode)
pip install -e spacy_uzbek/
```

Verify:

```powershell
python -c "import spacy; nlp = spacy.blank('uz'); print(nlp.lang)"
# Output: uz
```

#### 2. GPU support (CUDA 12.x)

For GPU-accelerated training, install CuPy. **Pin to `13.6.0`** — CuPy ≥ 14.x breaks DLPack interoperability with PyTorch 2.6 and causes a `RuntimeError: from_dlpack received an invalid capsule` crash during backprop:

```powershell
pip install cupy-cuda12x==13.6.0
```

Verify GPU is accessible:

```powershell
python -c "import spacy; spacy.require_gpu(0); print('GPU OK')"
```

#### 3. Convert CoNLL-U data to spaCy format

```bash
# Convert all available splits (UzUDT, UT, merged)
python spacy_uzbek/convert_conllu.py --convert-all

# Or convert a single file
python spacy_uzbek/convert_conllu.py \
    --input data/pos/uz_uzudt.train.in.conllu \
    --output spacy_uzbek/data/uz_uzudt.train.spacy
```

This produces `.spacy` (DocBin) files in `spacy_uzbek/data/` with UPOS, XPOS, morphological features, lemmas, and dependency annotations preserved.

#### 4. (Optional) Convert FastText vectors for spaCy

```bash
python -m spacy init vectors uz \
    wordvec/uz/fasttext/cc.uz.300.vec \
    wordvec/uz/spacy_vectors \
    --truncate 50000 --name uz_fasttext_vectors
```

### Training Configs

Three configurations are provided in `spacy_uzbek/configs/`:

| Config | File | Embeddings | GPU Required | W&B run name |
|--------|------|------------|--------------|-------------|
| **Transformer** | `config_transformer.cfg` | TahrirchiBERT (768-dim) | Recommended | `transformer_combined` |
| **Static** | `config_static.cfg` | Hash embeddings only | No | `static_combined` |
| **FastText** | `config_fasttext.cfg` | FastText (300-dim) | No | `fasttext_combined` |

Each config trains a joint pipeline: **tagger** (UPOS) + **morphologizer** (UFeats) + **parser** (UAS/LAS).

### Logging & Checkpoints

All three configs use `spacy.WandbLogger.v3` (project: **`spacy-uzbek`**), which streams metrics to W&B and also prints the progress table to the terminal. Make sure you have logged in once:

```powershell
wandb login
```

Every training run saves two checkpoints to the `--output` directory:

| Path | Contents |
|------|----------|
| `model-best/` | Weights at the epoch with the highest combined dev score |
| `model-last/` | Weights at the final training step |

Override the W&B run name per experiment with `--training.logger.run_name <name>` on the CLI (see examples below).

### Training Commands

> **PowerShell note:** Use backtick (`` ` ``) for line continuation. Bash-style backslash (`\`) does **not** work in PowerShell. To avoid issues, you can also paste all arguments on one line.

#### Transformer pipeline (TahrirchiBERT) — recommended

```powershell
# Merged treebanks (UzUDT + UT)
python -m spacy train spacy_uzbek/configs/config_transformer.cfg `
    --output saved_models/spacy/transformer_combined `
    --paths.train spacy_uzbek/data/uz_combined.train.spacy `
    --paths.dev spacy_uzbek/data/uz_combined.dev.spacy `
    --training.logger.run_name transformer_combined `
    --gpu-id 0

# UzUDT only
python -m spacy train spacy_uzbek/configs/config_transformer.cfg `
    --output saved_models/spacy/transformer_uzudt `
    --paths.train spacy_uzbek/data/uz_uzudt.train.spacy `
    --paths.dev spacy_uzbek/data/uz_uzudt.dev.spacy `
    --training.logger.run_name transformer_uzudt `
    --gpu-id 0
```

One-liner equivalents (no continuation characters):

```powershell
python -m spacy train spacy_uzbek/configs/config_transformer.cfg --output saved_models/spacy/transformer_combined --paths.train spacy_uzbek/data/uz_combined.train.spacy --paths.dev spacy_uzbek/data/uz_combined.dev.spacy --training.logger.run_name transformer_combined --gpu-id 0
```

#### Static baseline (CPU)

```powershell
python -m spacy train spacy_uzbek/configs/config_static.cfg `
    --output saved_models/spacy/static_combined `
    --paths.train spacy_uzbek/data/uz_combined.train.spacy `
    --paths.dev spacy_uzbek/data/uz_combined.dev.spacy `
    --training.logger.run_name static_combined
```

#### FastText vectors

```powershell
# First convert vectors (one-time):
python -m spacy init vectors uz `
    wordvec/uz/fasttext/cc.uz.300.vec `
    wordvec/uz/spacy_vectors `
    --truncate 50000 --name uz_fasttext_vectors

# Then train:
python -m spacy train spacy_uzbek/configs/config_fasttext.cfg `
    --output saved_models/spacy/fasttext_combined `
    --paths.train spacy_uzbek/data/uz_combined.train.spacy `
    --paths.dev spacy_uzbek/data/uz_combined.dev.spacy `
    --paths.vectors wordvec/uz/spacy_vectors `
    --training.logger.run_name fasttext_combined
```

### Evaluation

> **Windows note:** Always use `.venv\Scripts\python.exe` explicitly for `spacy evaluate` — the system `python` does not have CuPy installed and will fail with `ValueError: Cannot use GPU, CuPy is not installed`.

```powershell
# Create results directory (one-time)
New-Item -ItemType Directory -Force -Path results | Out-Null

# S1.1 — UzUDT model on UzUDT test set
.venv\Scripts\python.exe -m spacy evaluate `
    saved_models/spacy/transformer_uzudt/model-best `
    spacy_uzbek/data/uz_uzudt.test.spacy `
    --output results/spacy_s1.1_test.json --gpu-id 0

# S1.2 — Combined model on combined test set
.venv\Scripts\python.exe -m spacy evaluate `
    saved_models/spacy/transformer_combined/model-best `
    spacy_uzbek/data/uz_combined.test.spacy `
    --output results/spacy_s1.2_test.json --gpu-id 0
```

One-liner equivalents:

```powershell
.venv\Scripts\python.exe -m spacy evaluate saved_models/spacy/transformer_uzudt/model-best spacy_uzbek/data/uz_uzudt.test.spacy --output results/spacy_s1.1_test.json --gpu-id 0
.venv\Scripts\python.exe -m spacy evaluate saved_models/spacy/transformer_combined/model-best spacy_uzbek/data/uz_combined.test.spacy --output results/spacy_s1.2_test.json --gpu-id 0
```

### Using the Trained Model

```python
import spacy

# Load trained model
nlp = spacy.load("saved_models/spacy/transformer_combined/model-best")

# Process Uzbek text
doc = nlp("Men kitob o'qiyapman.")

for token in doc:
    print(f"{token.text:20s}  POS={token.pos_:8s}  DEP={token.dep_:12s}  HEAD={token.head.text}")

# Visualize dependency tree
from spacy import displacy
displacy.serve(doc, style="dep")
```

### Packaging for Distribution

```bash
# Package the trained model for distribution
python -m spacy package saved_models/spacy/transformer_combined/model-best \
    packages/ --name uz_pipeline --version 0.1.0

# Install the packaged model
pip install packages/uz_pipeline-0.1.0/dist/*.whl

# Use it anywhere
import spacy
nlp = spacy.load("uz_pipeline")
```

---

## Author

**Sanatbek Matlatipov**  
Researcher — National University of Uzbekistan  
s.matlatipov@nuu.uz

---

## Citation

```bibtex
@inproceedings{Matlatipov2025Robust,
  title     = {Towards Robust Uzbek Neural Dependency Parsing},
  author    = {Matlatipov, Sanatbek},
  booktitle = {Proceedings of the Conference (To Appear)},
  year      = {2025}
}
```

