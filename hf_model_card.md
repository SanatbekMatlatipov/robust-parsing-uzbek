---
language:
  - uz
license: cc-by-sa-4.0
library_name: stanza
tags:
  - uzbek
  - dependency-parsing
  - pos-tagging
  - nlp
  - low-resource
  - ud
  - stanza
  - bert
  - fasttext
datasets:
  - UD_Uzbek-UzUDT
  - UD_Uzbek-UT
metrics:
  - las
  - uas
  - accuracy
---

# Towards Robust Uzbek Neural Dependency Parsing — Model Weights

This repository hosts trained model checkpoints from the paper  
**"Towards Robust Uzbek Neural Dependency Parsing"** (Matlatipov, 2026).

The models are **Stanza-style** neural pipelines for Uzbek morphosyntactic tagging (UPOS/XPOS/UFeats)
and UD dependency parsing (UAS/LAS), comparing a **static FastText baseline** against
**TahrirchiBERT** contextual embeddings across two Uzbek UD treebanks.

Source code & training scripts: https://github.com/Sanatbek/robust-parsing-uzbek

---

## Model Files

### Tokenizer

| File | Description |
|------|-------------|
| `saved_models/tokenize/uz_uzudt_tokenizer.pt` | Uzbek tokenizer trained on UzUDT |

### POS Taggers (`saved_models/pos/`)

| File | Experiment | Data | Embeddings | Fusion |
|------|------------|------|------------|--------|
| `uz_uzudt_E1_tagger.pt` | E1 baseline | UzUDT | FastText | — |
| `uz_uzudt_E2.1_tagger.pt` | E2.1 | UzUDT | TahrirchiBERT | last-subword |
| `uz_uzudt_E3.1_tagger.pt` | E3.1 | UzUDT | TahrirchiBERT | mean pooling |
| `uz_uzudt_E5.1_tagger.pt` | E5.1 | UzUDT | TahrirchiBERT + charlm | last-subword |
| `uz_uzudt_E5.1.1_tagger.pt` | E5.1.1 | UzUDT | TahrirchiBERT + charlm (ablation) | last-subword |
| `uz_uzudt-base_tagger.pt` | Base | UzUDT | TahrirchiBERT | last-subword |
| `uz_combined_E1.2_tagger.pt` | E1.2 | UzUDT+UT | FastText | — |
| `uz_combined_E2.2_tagger.pt` | E2.2 | UzUDT+UT | TahrirchiBERT | last-subword |
| `uz_combined_E3.2_tagger.pt` | E3.2 | UzUDT+UT | TahrirchiBERT | mean pooling |

### Dependency Parsers (`saved_models/depparse/`)

| File | Experiment | Data | Embeddings | Fusion |
|------|------------|------|------------|--------|
| `uz_uzudt_E1.1_parser.pt` | E1.1 | UzUDT | FastText | — |
| `uz_uzudt_E2.1_parser.pt` | E2.1 | UzUDT | TahrirchiBERT | last-subword |
| `uz_uzudt_E3.1_parser.pt` | E3.1 | UzUDT | TahrirchiBERT | mean pooling |
| `uz_uzudt_E5.1_parser.pt` | E5.1 | UzUDT | TahrirchiBERT + charlm | last-subword |
| `uz_uzudt_nocharlm_parser.pt` | Ablation | UzUDT | TahrirchiBERT (no charlm) | last-subword |
| `uz_combined_E1.2_parser.pt` | E1.2 | UzUDT+UT | FastText | — |
| `uz_combined_E2.2_parser.pt` | E2.2 | UzUDT+UT | TahrirchiBERT | last-subword |
| `uz_combined_E3.2_parser.pt` | E3.2 | UzUDT+UT | TahrirchiBERT | mean pooling |

---

## Evaluation Results (Test Set)

| Exp | Data | Embeddings | Fusion | UPOS | XPOS | UFeats | UAS | LAS |
|-----|------|------------|--------|------|------|--------|-----|-----|
| E1.1 | UzUDT | FastText | — | 79.19 | 79.81 | 66.61 | 69.57 | 51.24 |
| E1.2 | UzUDT+UT | FastText | — | 80.26 | 83.20 | 66.98 | 72.27 | 62.40 |
| E2.1 | UzUDT | TahrirchiBERT | last-sub | 82.45 | 80.90 | 65.37 | 72.05 | 54.19 |
| **E2.2** | **UzUDT+UT** | **TahrirchiBERT** | **last-sub** | **85.08** | **84.72** | **71.09** | **72.39** | **63.81** |
| E3.1 | UzUDT | TahrirchiBERT | mean | 82.76 | 81.37 | 65.22 | 69.10 | 51.55 |
| E3.2 | UzUDT+UT | TahrirchiBERT | mean | 84.02 | **87.07** | 70.39 | 70.74 | 60.05 |

**Best overall system: E2.2** — TahrirchiBERT + last-subword + merged data.

---

## How to Use

### 1. Clone this code repository

```bash
git clone https://github.com/Sanatbek/robust-parsing-uzbek.git
cd robust-parsing-uzbek
```

### 2. Set up environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install -e stanza/
```

### 3. Download models from this HuggingFace repository

Install the HuggingFace Hub client if not already present:

```bash
pip install huggingface_hub
```

Download all models at once:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Sanatbek/uzudt",
    repo_type="model",
    local_dir=".",
    ignore_patterns=["*.md", ".gitattributes"]
)
```

Or download a specific model:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Sanatbek/uzudt",
    filename="saved_models/pos/uz_combined_E2.2_tagger.pt",
    local_dir="."
)
```

### 4. Run POS-only inference

```bash
python scripts/parse_test_pos_only.py \
  --tagger_model saved_models/pos/uz_combined_E2.2_tagger.pt \
  --input_file data/pos/uz_uzudt.test.in.conllu \
  --output_file output_pos.conllu
```

### 5. Run full pipeline (POS + dependency parsing)

**FastText baseline (E1.2):**
```bash
python scripts/parse_test_with_depparse.py \
  --tagger_model saved_models/pos/uz_combined_E1.2_tagger.pt \
  --parser_model saved_models/depparse/uz_combined_E1.2_parser.pt \
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt \
  --input_file data/depparse/uz_uzudt.test.in.conllu \
  --output_file output_e1.conllu
```

**Best BERT model (E2.2 — recommended):**
```bash
python scripts/parse_test_with_depparse.py \
  --tagger_model saved_models/pos/uz_combined_E2.2_tagger.pt \
  --parser_model saved_models/depparse/uz_combined_E2.2_parser.pt \
  --bert_model tahrirchi/tahrirchi-bert-base \
  --input_file data/depparse/uz_uzudt.test.in.conllu \
  --output_file output_e2.conllu
```

### 6. Evaluate

```bash
# UD metrics (UAS, LAS, CLAS, MLAS, BLEX)
python scripts/eval.py \
  data/depparse/uz_uzudt.test.in.conllu \
  output_e2.conllu

# POS accuracy
python scripts/eval_pos.py \
  --gold data/pos/uz_uzudt.test.in.conllu \
  --system output_pos.conllu
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.9 | Runtime |
| PyTorch | >= 2.0 | Model inference |
| transformers | >= 4.35 | TahrirchiBERT loading |
| stanza | local (editable) | NLP pipeline |
| huggingface_hub | >= 0.20 | Model download |

---

## Citation

If you use these models, please cite:

```bibtex
@misc{matlatipov2026uzbek,
  title   = {Towards Robust Uzbek Neural Dependency Parsing},
  author  = {Matlatipov, Sanatbek},
  year    = {2026},
  url     = {https://huggingface.co/Sanatbek/uzudt}
}
```

---

## License

CC BY-SA 4.0 — see [LICENSE](LICENSE).
