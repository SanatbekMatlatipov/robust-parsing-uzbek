# Research Log — Towards Robust Uzbek Neural Dependency Parsing

**Author:** Sanatbek Matlatipov  
**Contact:** s.matlatipov@nuu.uz  
**Institution:** National University of Uzbekistan  
**Last Updated:** 2026-02-24

---

## 1. Project Overview

This project develops and evaluates a robust neural dependency parsing pipeline for Uzbek — a low-resource, agglutinative Turkic language. The key contributions are:

- A new manually annotated UD treebank for Uzbek (**UD_Uzbek-UzUDT**), larger and more diverse than any prior resource.
- A Stanza-based neural pipeline enhanced with **BERTbek** contextual embeddings and **Apertium**-based morphological normalization.
- Systematic benchmarking against baselines (UDPipe, spaCy) on a standardized test set.

---

## 2. Datasets

### 2.1 UD_Uzbek-UzUDT *(authored by Sanatbek Matlatipov)*

| Property | Detail |
|----------|--------|
| Source | Uzbek literature and educational writing |
| Sentences | 684 (total) |
| Tokens | ~7,800 |
| Annotation platform | INCEpTION |
| Annotators | 6 (4 linguists + 2 NLP engineers) |
| Cross-verification | Full adjudication |
| Inter-annotator agreement | >95% lemma, ~95% UPOS, ~90% morphfeats (Cohen's κ, Krippendorff's α) |
| Annotation layers | UPOS, XPOS, lemma, morphological features, dependency relations |
| UD guidelines | v2 |
| License | CC BY-SA 4.0 |
| UD release | v2.17 (initial: 2025-10-01) |
| Contributors | Matlatipov, Sanatbek; Kuriyozov, Elmurod |
| Genre | Fiction, Academic |

**Data Splits (final, as of 2026-02-24):**

| Split | File | Sentences | % |
|-------|------|-----------|---|
| Train | `uz_uzudt-ud-train.conllu` | 451 | 65.9% |
| Dev   | `uz_uzudt-ud-dev.conllu`   | 45  | 6.6%  |
| Test  | `uz_uzudt-ud-test.conllu`  | 188 | 27.5% |
| **Total** | | **684** | |

> **Split rationale:** The original repository contained only a train (483) and test (201) file — no dev split. Both files were pooled (684 sentences) and re-split using the standard UD proportions ~66% / ~7% / ~27% to create a proper train/dev/test partition. The proportions were derived from the UD community standard (approximately matching the 483/48/201 ratio of the prior annotated version).

---

### 2.2 UD_Uzbek-UT *(authored by Akhundjanova, Arofat and Talamo, Luigi)*

| Property | Detail |
|----------|--------|
| Source | News articles (250 sentences) + Fiction (250 sentences) |
| Sentences | 500 (total) |
| Tokens | ~5,850 |
| Annotation | Semi-automatic with full manual correction |
| Tokenization / Lemmatization | Automatic |
| POS & Dependency | Semi-automatic + manual |
| License | CC BY-SA 4.0 |
| UD release | v2.15 (initial: 2024-11-15) |
| Contributors | Akhundjanova, Arofat |
| Genre | News, Fiction |

**Reference:**
```bibtex
@inproceedings{akhundjanova-talamo-2025-universal,
    title     = "{U}niversal {D}ependencies Treebank for {U}zbek",
    author    = "Akhundjanova, Arofat and Talamo, Luigi",
    booktitle = "Proceedings of the Third Workshop on Resources and Representations for Under-Resourced Languages and Domains (RESOURCEFUL-2025)",
    month     = mar,
    year      = "2025",
    address   = "Tallinn, Estonia",
    publisher = "University of Tartu Library, Estonia",
    url       = "https://aclanthology.org/2025.resourceful-1.1/",
    pages     = "1--6"
}
```

**Data Splits (final, as of 2026-02-24):**

| Split | File | Sentences | % |
|-------|------|-----------|---|
| Train | `uz_ut-ud-train.conllu` | 330 | 66.0% |
| Dev   | `uz_ut-ud-dev.conllu`   | 33  | 6.6%  |
| Test  | `uz_ut-ud-test.conllu`  | 137 | 27.4% |
| **Total** | | **500** | |

> **Split rationale:** The original release contained only `uz_ut-ud-test.conllu` with all 500 sentences (no train/dev split). The file was split sequentially using the same ~66% / ~7% / ~27% proportions as UzUDT. The first 330 sentences are news-genre, the last 170 are fiction; so train is predominantly news, test is predominantly fiction — which is a realistic cross-genre evaluation scenario worth noting in the paper.

---

## 3. Pipeline Architecture

```
Raw Uzbek Text
       │
       ▼
┌─────────────────────────┐
│  Apertium Normalization │  ← morphological normalization, reduces lexical sparsity
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   BERTbek Encoder       │  ← elmurod1202/bertbek-news-big-cased
│  (subword tokenization) │
└────────────┬────────────┘
             │  "super-token" fusion: subwords → UD word tokens
             ▼
┌─────────────────────────┐
│  Joint Tagger + Parser  │  ← DeepBiaffine (Stanza-based)
│  UPOS / XPOS / Feats    │
│  Dependency arcs + rels │
└─────────────────────────┘
```

### Components

| Component | Detail |
|-----------|--------|
| Framework | Stanza (modified) |
| Embeddings | BERTbek (`elmurod1202/bertbek-news-big-cased`) |
| Fallback vectors | FastText `cc.uz.300.vec` (300-dim) |
| Parser | DeepBiaffine graph-based |
| Normalization | Apertium-based morphological pipeline |
| Subword fusion | Mean pooling of BERT subwords per UD token ("super-token") |
| Pretrained models | Hosted at [huggingface.co/Sanatbek/uzudt](https://huggingface.co/Sanatbek/uzudt) |

---

## 4. Saved Models

| Model | File | Location |
|-------|------|----------|
| POS Tagger | `uz_uzudt-base_tagger.pt` | `saved_models/pos/` |
| Dependency Parser | `uz_uzudt_nocharlm_parser.pt` | `saved_models/depparse/` |
| Tokenizer | `uz_uzudt_tokenizer.pt` | `saved_models/tokenize/` |

All models are also available on Hugging Face: `Sanatbek/uzudt`

---

## 5. Key Findings from Code Review

| Finding | Detail |
|---------|--------|
| **POS/depparse have NO BERT support** | The Stanza POS tagger (`tagger.py`) and dependency parser (`parser.py`) only support static pretrained embeddings via `--wordvec_pretrain_file`. Only the NER tagger and constituency parser have `--bert_model`. |
| **No Apertium code in repo** | The paper describes Apertium normalization but no implementation exists in the codebase. |
| **xpos_vocab_factory missing Uzbek** | `config/xpos_vocab_factory.py` does not list `uz_uzudt` — will throw `NotImplementedError`. |
| **Super-token fusion mismatch** | Paper says "mean pooling"; code in `bert_embedding.py` uses "last-subword selection" (not mean). |
| **Hyperparameter mismatch** | Paper Table 2: AdamW, lr=2e-5, batch=32, epochs=30. Stanza defaults: Adam, lr=3e-3, batch=5000 tokens, max_steps=50000. |
| **Previous model used XLM-RoBERTa** | The previous tagger was `uz_uzudt_xlm-roberta-base_tagger.pt`, not BERTbek as described in the paper. |
| **LAS discrepancy** | Paper reports LAS=66.90, but the actual log shows LAS=53.21. The 66.90 figure may come from a different checkpoint or evaluation configuration. |

---

## 6. Retraining Plan (2026-02-24)

### 6.1 Objectives

Retrain the full Stanza pipeline from scratch with proper BERT integration into POS tagger and dependency parser, testing multiple configurations to find the best system. The goal is to produce paper-ready, reproducible results.

### 6.2 Transformer Models to Test

| ID | Model | HuggingFace Hub | Hidden Size | Notes |
|----|-------|-----------------|-------------|-------|
| **BERT-A** | TahrirchiBERT | `tahrirchi/tahrirchi-bert-base` | 768 | Uzbek monolingual BERT, potentially broader domain coverage |
| **BERT-B** | BERTbek | `elmurod1202/bertbek-news-big-cased` | 768 | Uzbek monolingual BERT, news-domain pretrained, cased |

### 6.3 Experimental Matrix

The following experiments will be run on **both treebanks** (UD_Uzbek-UzUDT and UD_Uzbek-UT):

| Exp | Transformer | Apertium Norm | Super-token Fusion | Static Pretrain | Description |
|-----|-------------|---------------|-------------------|-----------------|-------------|
| E1 | None | No | N/A | FastText cc.uz.300 | Pure FastText baseline (no BERT) |
| E2 | None | Yes | N/A | FastText cc.uz.300 | FastText + Apertium normalization |
| E3 | TahrirchiBERT | No | Last-subword (current code) | None | TahrirchiBERT, no pooling change, no norm |
| E4 | TahrirchiBERT | No | Mean pooling | None | TahrirchiBERT, mean pooling |
| E5 | TahrirchiBERT | Yes | Last-subword | None | TahrirchiBERT + Apertium, last-subword |
| E6 | TahrirchiBERT | Yes | Mean pooling | None | TahrirchiBERT + Apertium, mean pooling |
| E7 | BERTbek | No | Last-subword | None | BERTbek, no pooling change, no norm |
| E8 | BERTbek | No | Mean pooling | None | BERTbek, mean pooling |
| E9 | BERTbek | Yes | Last-subword | None | BERTbek + Apertium, last-subword |
| E10 | BERTbek | Yes | Mean pooling | None | BERTbek + Apertium, mean pooling |
| E11 | TahrirchiBERT | Yes | Mean pooling | FastText cc.uz.300 | TahrirchiBERT + Apertium + FastText (combined) |
| E12 | BERTbek | Yes | Mean pooling | FastText cc.uz.300 | BERTbek + Apertium + FastText (combined) |

For each experiment: train POS tagger → re-tag data → train dependency parser → evaluate on test.

### 6.4 Code Changes Required

#### 6.4.1 Add `--bert_model` to POS Tagger and Parser

**Files to modify:**
- `stanza/stanza/models/tagger.py` — add `--bert_model` argument to `parse_args()`
- `stanza/stanza/models/parser.py` — add `--bert_model` argument to `parse_args()`
- `stanza/stanza/models/pos/model.py` — accept `bert_model` / `bert_tokenizer` in `__init__()`, add `bert_model.config.hidden_size` to `input_size`, call `extract_bert_embeddings()` in `forward()`
- `stanza/stanza/models/depparse/model.py` — same pattern as POS model
- `stanza/stanza/models/pos/trainer.py` — pass `bert_model`/`bert_tokenizer` to `Tagger()` constructor, pass raw sentences through batch
- `stanza/stanza/models/depparse/trainer.py` — same pattern as POS trainer
- `stanza/stanza/models/pos/data.py` — include raw text strings in batch output
- `stanza/stanza/models/depparse/data.py` — include raw text strings in batch output

**Template to follow:** The NER model (`stanza/stanza/models/ner/model.py`) already integrates BERT using this exact pattern.

#### 6.4.2 Implement Mean Pooling in `bert_embedding.py`

**File:** `stanza/stanza/models/common/bert_embedding.py`

Current `extract_bert_embeddings()` uses last-subword selection:
```python
list_offsets[idx][offset+1] = pos  # overwrites, keeping last subword position
```

Need to add a `pooling_strategy` parameter (`"last"` | `"mean"` | `"first"`):
- `"last"`: current behavior (take last subword per word)
- `"mean"`: average all subwords mapped to each word
- `"first"`: take first subword per word (like PhoBERT does)

#### 6.4.3 Fix xpos_vocab_factory

**File:** `config/xpos_vocab_factory.py`

Add `uz_uzudt` and `uz_ut` to the appropriate branch. Based on the treebank data (XPOS uses single-letter tags like `N`, `V`, `A`, `P`, etc.), Uzbek should map to:
```python
return WordVocab(data, shorthand, idx=2, ignore=["_"])
```

#### 6.4.4 Implement Apertium Normalization

**New file:** `scripts/apertium_normalize.py` or `stanza/stanza/utils/apertium.py`

The Apertium Uzbek package (`apertium-uzb`) provides:
- Lexicon lookup
- Morphotactic analysis
- Lemma normalization

This needs to be applied as a preprocessing step on CoNLL-U files BEFORE training:
1. Read CoNLL-U sentences
2. For each token, run through Apertium morphological analysis
3. Replace/normalize the LEMMA field based on Apertium output
4. Optionally populate/correct FEATS field
5. Write normalized CoNLL-U files

Two modes needed:
- **Preprocessing mode:** Normalize train/dev/test CoNLL-U files before training
- **Pipeline mode:** Normalize raw text at inference time before passing to the model

### 6.5 Implementation Order

1. **Fix `xpos_vocab_factory.py`** — add Uzbek shorthand (quick fix)
2. **Add BERT to POS model** — modify model.py, trainer.py, data.py, tagger.py (follow NER template)
3. **Add BERT to depparse model** — same pattern
4. **Implement mean pooling option** in `bert_embedding.py`
5. **Implement Apertium normalization script** — preprocessing CoNLL-U files
6. **Run E1–E2** (FastText baselines) — verify pipeline works end-to-end
7. **Run E3–E6** (TahrirchiBERT experiments)
8. **Run E7–E10** (BERTbek experiments)
9. **Run E11–E12** (combined BERT + FastText)
10. **Collect results, update RESEARCH_LOG, write paper tables**

---

## 7. Key Decisions & Notes for Paper

1. **Why BERTbek over multilingual BERT?** Monolingual Uzbek pretraining captures morphological patterns of agglutination better than mBERT, yielding ~+10 LAS over the multilingual baseline.

2. **Why also test TahrirchiBERT?** `tahrirchi/tahrirchi-bert-base` is another monolingual Uzbek BERT that may have broader domain coverage. Testing both enables a fair comparison in the paper.

3. **Why Apertium normalization?** Uzbek orthography inconsistencies (Cyrillic vs. Latin script mixing, diacritics) cause significant out-of-vocabulary issues. Apertium preprocessing reduces token-level sparsity.

4. **Super-token fusion ablation:** The paper describes mean pooling, but the Stanza code uses last-subword selection. We will test both to determine which is better — this becomes an ablation study contribution.

5. **UzUDT vs. UD_Uzbek-UT:** UzUDT (your treebank) is manually and fully annotated from scratch, covers literature/academic text, and is larger (684 sentences). UD_Uzbek-UT (Akhundjanova & Talamo, 2025) covers news+fiction and used semi-automatic annotation. The two treebanks are complementary and can be used for cross-corpus evaluation.

6. **Dev split creation:** Neither treebank originally had a dev set in a ready-to-use form. Dev splits were carved from the full data using consistent 66/7/27 proportions to enable standard training practices.

7. **Genre bias in UD_Uzbek-UT splits:** Because the UT corpus is ordered news-first (sentences 1–250) then fiction (251–500), the sequential split means train ≈ news-heavy, test ≈ fiction-heavy. This is worth acknowledging in cross-corpus experiments.

---

## 8. Changelog

| Date | Action |
|------|--------|
| 2026-02-24 | Full code review: identified BERT not integrated in POS/depparse, no Apertium code, xpos_vocab missing Uzbek |
| 2026-02-24 | Defined 12-experiment retraining plan with TahrirchiBERT + BERTbek + Apertium + fusion ablation |
| 2026-02-24 | Created `RESEARCH_LOG.md` |
| 2026-02-24 | Pooled UD_Uzbek-UzUDT train+test (684 sents) and re-split into train/dev/test (451/45/188) |
| 2026-02-24 | Split UD_Uzbek-UT all-data file (500 sents) into train/dev/test (330/33/137) |
| 2025-11-07 | Previous training: POS tagger with XLM-RoBERTa-base + FastText, parser with FastText only |
| 2025-10-01 | UD_Uzbek-UzUDT initial UD v2.17 release |
| 2024-11-15 | UD_Uzbek-UT initial UD v2.15 release (Akhundjanova & Talamo) |
