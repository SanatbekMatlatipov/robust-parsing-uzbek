Here is the updated `README.md`. It integrates your **repository structure**, points to the **Hugging Face models**, updates the **performance metrics** to match your paper, and explains the **methodology** (BERTbek + Apertium) described in your research.


# Towards Robust Uzbek Neural Dependency Parsing

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/Sanatbek/uzudt)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This repository contains the **official implementation, evaluation pipelines, and scripts** for the paper *Towards Robust Uzbek Neural Dependency Parsing*.

It provides a framework for training and evaluating **Stanza-style neural pipelines** for Uzbek, featuring:
* **BERTbek Contextual Embeddings** (replacing standard word vectors for robustness).
* **Morphology-Aware Preprocessing** (integrating Apertium normalization).
* **UD Benchmarking** on the 3-star [UzUDT Treebank](https://github.com/UniversalDependencies/UD_Uzbek-UzUDT).

**🔗 Pretrained Models:** The trained models (Tokenizer, Tagger, Parser) are hosted on Hugging Face: [huggingface.co/Sanatbek/uzudt](https://huggingface.co/Sanatbek/uzudt).


## 🗂 Repository Structure
```markdown
uzudtevaluations/
│
├── scripts/
│   ├── pos_predict_from_udbase.py     # Script to generate POS predictions
│   ├── eval_pos.py                    # Standard POS evaluation
│   ├── eval_upos_by_tag.py            # Granular error analysis by tag
│   └── eval.py                        # CoNLL 2018 shared task evaluation script
├── data/
│   └── udbase/
│       └── UD_Uzbek-UzUDT/            # The 3-star UzUDT corpus
│           ├── uz_uzudt-ud-train.conllu
│           ├── uz_uzudt-ud-dev.conllu
│           └── uz_uzudt-ud-test.conllu
├── wordvec/
│   └── uz/
│       └── cc.uz.300.vec              # FastText vectors (fallback/baseline)
├── saved_models/
│   ├── pos/
│   │   └── uz_uzudt-base_tagger.pt    # PyTorch model for POS tagging
│   └── depparse/
│       ├── uz_uzudt_nocharlm_parser.pt
│       └── uz_uzudt_nocharlm_parser_checkpoint.pt
├── logs/
└── README.md

```

> **Note:** > While this repository structure supports `cc.uz.300.vec` (fastText), the primary results in the paper were achieved using **BERTbek** (`elmurod1202/bertbek-news-big-cased`). The `.pt` files in `saved_models/` utilize these transformer embeddings.

---

## ⚙️ Methodology

This system addresses the challenges of **agglutination** and **data scarcity** in Uzbek by employing a robust pipeline:

1. **Preprocessing:** Text normalization and Apertium-based morphological analysis to stabilize lemmas.
2. **Contextual Embedding:** Uses **BERTbek** with a subword-to-word "super-token" fusion strategy to align BERT subwords with UD tokens.
3. **Parsing:** A biaffine graph-based parser (DeepBiaffine) trained on the UzUDT treebank.

---

## 🧠 Setup Instructions

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
# Install core dependencies including Stanza and Transformers
pip install stanza transformers spacy conllu pandas numpy scikit-learn tqdm huggingface_hub

```

### 2. Download Pretrained Models

You can download the specific `.pt` checkpoints referenced in this repo directly from Hugging Face:

```bash
# Ensure you are in the root directory
mkdir -p saved_models/pos saved_models/depparse

# Download models (example using huggingface-cli)
huggingface-cli download Sanatbek/uzudt uz_uzudt-base_tagger.pt --local-dir saved_models/pos
huggingface-cli download Sanatbek/uzudt uz_uzudt_nocharlm_parser.pt --local-dir saved_models/depparse

```

---

## 🚀 Workflows

### A. POS Tagging (Stanza)

#### 1. Predict POS Tags

Run the predictor using the trained model in `saved_models/pos/`:

```bash
cd stanza-train/scripts
python3 pos_predict_from_udbase.py

```

*Outputs:* `data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.pos.system.conllu`

#### 2. Evaluate Accuracy

Calculate standard accuracy and granular tag-level performance:

```bash
python3 eval_pos.py \
  --gold ../data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.conllu \
  --system ../data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.pos.system.conllu

python3 eval_upos_by_tag.py \
  --gold ../data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.conllu \
  --system ../data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.pos.system.conllu

```

### B. Dependency Parsing (Stanza)

Run the full parsing pipeline (tokenization + tagging + parsing) on the test set:

```bash
python scripts/parse_test_with_depparse.py

```

### C. Baselines (spaCy & UDPipe)

This repository also contains scripts to train and evaluate baseline models for comparison.

* **spaCy:** Located in the `spacy/` directory (uses cross-lingual transfer from Turkish).
* **UDPipe:** Standard baseline comparison.

---

## 📊 Performance

The following results are reported on the **UzUDT Test Set** (681 sentences), establishing the current State-of-the-Art for Uzbek dependency parsing.

| Metric | Description | Score (%) |
| --- | --- | --- |
| **UPOS** | Universal POS Tagging Accuracy | **86.10** |
| **XPOS** | Language-Specific POS Accuracy | **83.96** |
| **UAS** | Unlabeled Attachment Score | **74.21** |
| **LAS** | Labeled Attachment Score | **66.90** |
| **UFeats** | Morphological Features Accuracy | **70.06** |

> **Comparison:** The Stanza+BERTbek pipeline significantly outperforms baselines (UDPipe LAS ~45.0, spaCy LAS ~51.0) by leveraging monolingual contextual embeddings and robust preprocessing.

---

## 👤 Author

**Sanatbek Matlatipov** Researcher – National University of Uzbekistan

📧 s.matlatipov@nuu.uz

---

## 📄 Citation

If you use this code or the models in your research, please cite the paper:

```bibtex
@inproceedings{Matlatipov2025Robust,
  title={Towards Robust Uzbek Neural Dependency Parsing},
  author={Matlatipov, Sanatbek},
  booktitle={Proceedings of the Conference (To Appear)},
  year={2025}
}

```
