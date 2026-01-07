# 🇺🇿 UzUDT Evaluations

This repository contains **evaluation pipelines, scripts, and resources** for the **Uzbek Universal Dependencies Treebank (UzUDT)** and the **neural dependency parsing experiments**.

The repo supports **training, prediction, and evaluation** of POS tagging and dependency parsing models using:
- [Stanza](https://stanfordnlp.github.io/stanza/)
- [spaCy](https://spacy.io/)
- Baseline systems such as **UDPipe**
- Custom Uzbek-trained models and corpora

---

## 🗂 Repository Structure

```
uzudtevaluations/
│
├── scripts/
│   ├── pos_predict_from_udbase.py
│   ├── eval_pos.py
│   └── eval_upos_by_tag.py
├── data/
│   └── udbase/
│       └── UD_Uzbek-UzUDT/
│           ├── uz_uzudt-ud-train.conllu
│           ├── uz_uzudt-ud-dev.conllu
│           └── uz_uzudt-ud-test.conllu
├── wordvec/
│   └── uz/
│       └── cc.uz.300.vec  ← downloaded Uzbek fastText word vectors
├── saved_models/
│   └── pos/
│       └── uz_uzudt-base_tagger.pt
│    └── depparse/
│      └── uz_uzudt_nocharlm_parser.pt
│      └── uz_uzudt_nocharlm_parser_checkpoint.pt
└── logs/
└── README.md
```
> 🧩 **Note:**  
> The `wordvec/uz/` directory must contain the Uzbek **fastText word embeddings** file  
> (`cc.uz.300.vec`), downloadable from [https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uz.300.vec.gz).  
> These pretrained vectors are required when training or fine-tuning Stanza POS or dependency models.

---

## ⚙️ Overview

This repository provides a **complete evaluation framework** for Uzbek NLP tasks, including:
- **POS tagging**
- **Dependency parsing**
- **UPOS-level accuracy and tag breakdown**
- **Cross-parser comparison (Stanza, spaCy, UDPipe)**

The experiments are based on the **UzUDT Treebank** — a manually annotated Universal Dependencies resource for Uzbek (≈7.8K tokens, 686 sentences).

---

## 🧠 Setup Instructions

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install stanza spacy conllu pandas numpy scikit-learn tqdm
```

### 2. spaCy Model (Proxy for Uzbek)

Because no native Uzbek model exists yet, the **Turkish spaCy model** serves as a linguistic base:

```bash
python -m spacy download tr_core_news_sm
```

---

## 🚀 Workflows

### A. POS Tagging (Stanza)

#### 1. Predict POS Tags
```bash
cd stanza-train/scripts
python3 pos_predict_from_udbase.py
```

Outputs:
```
data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.pos.system.conllu
```

The trained POS model is stored at:
```
stanza-train/saved_models/pos/uz_uzudt_xlm-roberta-base_tagger.pt
```

#### 2. Evaluate POS Accuracy
```bash
python3 eval_pos.py   --gold data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.conllu   --system data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.pos.system.conllu
```

#### 3. Evaluate by Tag
```bash
python3 eval_upos_by_tag.py   --gold data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.conllu   --system data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.pos.system.conllu
```

---

### B. Dependency Parsing (spaCy)

Train using Turkish as source model:

```bash
cd spacy
python train_uzbek_parser.py   --train ../data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-train.conllu   --dev ../data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-dev.conllu   --lang tr
```

Evaluate:
```bash
python evaluate_spacy_parser.py   --model output/model-best   --test ../data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.conllu
```
---

### 🧪 C. Model Testing & Evaluation (Stanza)

Once you have trained models for **tokenization**, **POS tagging**, and **dependency parsing**, you can evaluate their performance on the Uzbek UD test set.

#### 1️⃣ Parse the Test Set

Run your trained Stanza models (tokenizer + POS + parser) on the test set:
```bash
cd stanza-train
python scripts/parse_test_with_depparse.py
```
---
## 🧠 D. spaCy Parser for Uzbek 

In addition to Stanza, this project provides an experimental **spaCy** pipeline for Uzbek UD parsing.  
The spaCy setup lives at the project root level:

```text
uzudtevaluations/
  data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-*.conllu
  spacy/
    train_uzbek_parser.py
    evaluate_spacy_parser.py
    models/
    results/
 ```

## 📊 Parser Performance

The following table summarizes parsing and tagging accuracies (%) on the **new (Educational/Literary)** and **old (Uzbek-UT)** Uzbek treebanks.

| **Parser** | **LAS (New)** | **UAS (New)** | **UPOS (New)** | **LAS (Old)** | **UAS (Old)** | **UPOS (Old)** |
|-------------|----------------|----------------|----------------|----------------|----------------|----------------|
| UDPipe | 45.0 | 55.0 | 75.0 | 41.0 | 48.0 | 70.0 |
| spaCy | 51.0 | 60.0 | 80.0 | 48.0 | 56.0 | 78.0 |
| **Stanza** | **56.0** | **65.0** | **84.0** | **52.0** | **60.0** | **81.0** |

📘 **Interpretation:**
- Stanza consistently outperforms spaCy and UDPipe in all metrics.
- The educational/literary corpus yields higher accuracy than the legacy Uzbek-UT dataset.
- UPOS tagging accuracy remains the highest metric due to strong lexical generalization.

---

## 🧮 Metrics

| Metric | Description |
|---------|-------------|
| **LAS** | Labeled Attachment Score (correct head + label) |
| **UAS** | Unlabeled Attachment Score (correct head only) |
| **UPOS** | Universal Part-of-Speech tagging accuracy |

---

## 📚 References

1. Kübler, S., McDonald, R., & Nivre, J. (2009). *Dependency Parsing.* Morgan & Claypool Publishers.  
2. Matlatipov, S. G., et al. (2024). *UzUDT: Universal Dependencies Treebank for Uzbek.* National University of Uzbekistan.  
3. Nivre, J. et al. (2020). *Universal Dependencies v2: An evergrowing multilingual treebank collection.*

---

## 👤 Author

**Sanatbek Matlatipov**  
Researcher – National University of Uzbekistan
📧 s.matlatipov@nuu.uz

---

## 📄 License

Released under **CC BY-NC 4.0 License** (research and education only).

> **Citation:**
> ```bibtex
> @inproceedings{Matlatipov2025UzUDT,
>   title={UzUDT: ....},
>   author={Matlatipov, Sanatbek},
>   year={....},
>   booktitle={LREC–COLING 2025 Proceedings},
>   organization={ELRA}
> }
> ```
