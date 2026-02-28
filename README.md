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
│   └── depparse/                   # Trained parser checkpoints
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

All 4 experiment runs are complete. See `RESEARCH_LOG.md` for detailed analysis and training curves.

| Exp | Data | Embeddings | UPOS | XPOS | UFeats | UAS | LAS |
|-----|------|------------|------|------|--------|-----|-----|
| E1.1 | UzUDT | FastText | 79.19 | 79.81 | 66.61 | 69.57 | 51.24 |
| E1.2 | UzUDT+UT | FastText | 80.26 | 83.20 | 66.98 | 72.27 | 62.40 |
| E2.1 | UzUDT | TahrirchiBERT | 82.45 | 80.90 | 65.37 | 72.05 | 54.19 |
| E2.2 | UzUDT+UT | TahrirchiBERT | **85.08** | **84.72** | **71.09** | **72.39** | **63.81** |

### Key Findings

1. **TahrirchiBERT outperforms FastText on POS tagging.** The best BERT model (E2.2) achieves 85.08 UPOS vs 80.26 for the best FastText model (E1.2), a gain of **+4.82 points**.
2. **Merging treebanks consistently helps.** Every `.2` run (UzUDT+UT) outperforms its `.1` counterpart (UzUDT-only), with LAS gains of +11.16 (E1) and +9.62 (E2).
3. **Best system: E2.2** — TahrirchiBERT + UzUDT+UT merged data achieves the highest scores across all metrics (UPOS 85.08, LAS 63.81).

---

## Key Ablations

The experiments are designed to answer two questions:

1. **Does TahrirchiBERT improve over static FastText embeddings?** Compare E1 (FastText-only) vs E2 (TahrirchiBERT) — the answer is yes: +4.82 UPOS, +1.41 LAS on the merged data setting.
2. **Does merging two Uzbek UD treebanks help?** Compare `.1` (UzUDT-only) vs `.2` (UzUDT+UT merged) — the answer is yes: the combined data consistently yields higher scores across all metrics, especially LAS (+11.16 for E1, +9.62 for E2).

> **Future work:** Additional experiments with BERTbek, mean pooling, and BERT+FastText fusion are planned — see `future_research_log.md`.

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

