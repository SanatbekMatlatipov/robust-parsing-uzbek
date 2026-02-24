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

## 4. Results

### 4.1 Main Results (UzUDT Test Set)

| Metric | Description | Score (%) |
|--------|-------------|-----------|
| UPOS | Universal POS Accuracy | **86.10** |
| XPOS | Language-Specific POS Accuracy | **83.96** |
| UAS | Unlabeled Attachment Score | **74.21** |
| LAS | Labeled Attachment Score | **66.90** |
| UFeats | Morphological Features Accuracy | **70.06** |

### 4.2 Baseline Comparison

| System | LAS (%) |
|--------|---------|
| UDPipe | ~45.0 |
| spaCy (cross-lingual from Turkish) | ~51.0 |
| **Stanza + BERTbek (ours)** | **66.90** |

---

## 5. Saved Models

| Model | File | Location |
|-------|------|----------|
| POS Tagger | `uz_uzudt-base_tagger.pt` | `saved_models/pos/` |
| Dependency Parser | `uz_uzudt_nocharlm_parser.pt` | `saved_models/depparse/` |
| Tokenizer | `uz_uzudt_tokenizer.pt` | `saved_models/tokenize/` |

All models are also available on Hugging Face: `Sanatbek/uzudt`

---

## 6. Key Decisions & Notes for Paper

1. **Why BERTbek over multilingual BERT?** Monolingual Uzbek pretraining captures morphological patterns of agglutination better than mBERT, yielding ~+10 LAS over the multilingual baseline.

2. **Why Apertium normalization?** Uzbek orthography inconsistencies (Cyrillic vs. Latin script mixing, diacritics) cause significant out-of-vocabulary issues. Apertium preprocessing reduces token-level sparsity.

3. **UzUDT vs. UD_Uzbek-UT:** UzUDT (your treebank) is manually and fully annotated from scratch, covers literature/academic text, and is larger (684 sentences). UD_Uzbek-UT (Akhundjanova & Talamo, 2025) covers news+fiction and used semi-automatic annotation. The two treebanks are complementary and can be used for cross-corpus evaluation.

4. **Dev split creation:** Neither treebank originally had a dev set in a ready-to-use form. Dev splits were carved from the full data using consistent 66/7/27 proportions to enable standard training practices.

5. **Genre bias in UD_Uzbek-UT splits:** Because the UT corpus is ordered news-first (sentences 1–250) then fiction (251–500), the sequential split means train ≈ news-heavy, test ≈ fiction-heavy. This is worth acknowledging in cross-corpus experiments.

---

## 7. Changelog

| Date | Action |
|------|--------|
| 2026-02-24 | Created `RESEARCH_LOG.md` |
| 2026-02-24 | Pooled UD_Uzbek-UzUDT train+test (684 sents) and re-split into train/dev/test (451/45/188) |
| 2026-02-24 | Split UD_Uzbek-UT all-data file (500 sents) into train/dev/test (330/33/137) |
| Prior | Trained Stanza pipeline with BERTbek; achieved LAS 66.90 on UzUDT test |
| Prior | Added Apertium normalization layer to preprocessing |
| 2025-10-01 | UD_Uzbek-UzUDT initial UD v2.17 release |
| 2024-11-15 | UD_Uzbek-UT initial UD v2.15 release (Akhundjanova & Talamo) |
