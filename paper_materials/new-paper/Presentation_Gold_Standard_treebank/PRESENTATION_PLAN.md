# TurkLang 2026 — Presentation Plan (13 minutes)

**Paper:** A Gold-Standard Dependency Treebank for the Uzbek Language
**Authors:** S. Matlatipov, M. Kurbanova
**Venue:** XIV International Conference on Computer Processing of Turkic Languages — TurkLang 2026
**Time budget:** 13 minutes talk + ~2 minutes Q&A
**Focus (explicit user instruction):** the **dataset and its evaluations**, *not* parser-internal algorithms.

---

## 1. Time allocation

| # | Slide | Topic | Time | Cumulative |
|---|-------|-------|------|------------|
| 1 | Title | Title, authors, affiliation | 0:20 | 0:20 |
| 2 | Outline | Roadmap | 0:30 | 0:50 |
| 3 | Motivation | Why Uzbek? Low-resource Turkic, agglutinative SOV | 1:00 | 1:50 |
| 4 | Related work | Uzbek-UT (Akhundjanova & Talamo) + existing tools | 1:00 | 2:50 |
| 5 | Contributions | 681 sents, 6 annotators, 2 new domains, ↑LAS | 0:40 | 3:30 |
| 6 | Data collection | Sources: *Maqar*, *Kun shundan boshlanadi*, fairy tales | 1:00 | 4:30 |
| 7 | Annotation workflow | INCEpTION, double annotation, adjudication (Fig. 2) | 1:20 | 5:50 |
| 8 | Annotated example | TikZ dependency tree — Uzbek SOV with morphology | 1:30 | 7:20 |
| 9 | Inter-annotator agreement | Cohen κ / Krippendorff α table | 1:00 | 8:20 |
| 10 | Treebank statistics | Comparison vs. Uzbek-UT | 1:00 | 9:20 |
| 11 | Linguistic profile | POS / morph features / dep relations | 1:00 | 10:20 |
| 12 | Experimental setup | 3 parsers, splits — keep this *short*, high-level | 0:40 | 11:00 |
| 13 | Results | LAS / UAS / UPOS table + key finding | 1:20 | 12:20 |
| 14 | Error analysis | advcl, coordination, nmod/obl confusions | 0:30 | 12:50 |
| 15 | Conclusion | Take-aways + future work | 0:30 | 13:20 |
| 16 | Thank-you / Q&A | Acknowledgments, contact, repo link | — | — |

> 13:20 leaves a small safety margin; trim slides 4 or 12 if running long.

---

## 2. Key numbers to memorize

- **681 sentences**, **~7,950 tokens**, **~4,800 unique forms** (37% larger than Uzbek-UT)
- **6 annotators** = 4 linguists + 2 NLP engineers
- IAA: **κ = 95.2%** (lemma), **93.7%** (UPOS), **90.8%** (morph features)
- Splits: **481 train / 200 test** (ours); **400 / 100** (Uzbek-UT, genre-balanced)
- Best parser: **Stanza** — LAS **66.0**, UAS **74.0**, UPOS **97.0** on the new treebank
- LAS gains over Uzbek-UT: **+3 to +7.5 points** depending on parser
- UDPipe shows the **largest relative improvement** (~+16% LAS)

---

## 3. Visual assets

| Asset | Where | Status |
|-------|-------|--------|
| `annatation progress.png` | from paper folder | reuse on slide 7 |
| `fig1.png` (annotated example) | from paper folder | reuse on slide 8 (right column) |
| TikZ dependency tree (custom) | inline | included in `presentation.tex` |
| LDV logo (template) | `ldvtheme/` | keep in title/footer |

If you want polished originals, save updated versions into `paper_materials/review_images/`.

---

## 4. Speaker notes (per slide)

### Slide 3 — Motivation
- Uzbek = ~35M speakers, low NLP resourcing, complex agglutinative morphology, SOV order.
- Before this work, only one UD treebank existed (Uzbek-UT, 500 sents). Coverage was limited and parsing scores plateaued.

### Slide 7 — Annotation
- Stress *double annotation + adjudication*, NOT single-pass.
- Six annotators trained on a shared pilot. Senior linguist as final arbiter.
- Uzbek-specific decisions worth mentioning verbally:
  - Case suffixes → morphological features (`Case=Acc`, `Case=Dat`, …).
  - Postpositions → separate tokens with `case` relation.
  - Null copula in predicative constructions → *no* explicit `cop` node.

### Slide 8 — Example
- Walk through one Uzbek sentence; highlight `nsubj`, `obj`, `obl`, and the `Case=` features that the morphology drives.

### Slide 12 — Setup
- Keep parser internals out. One sentence each: UDPipe (pipeline baseline), Stanza (neural BiLSTM + char-CNN), spaCy (transition-based, tok2vec).
- Gold tokenization, default hyperparameters → fair comparison.

### Slide 13 — Results
- Lead with the headline: **Stanza 66 LAS / 97 UPOS** on the new treebank.
- Then the comparative claim: **every parser improves on the new treebank**.
- Frame this as evidence that the *resource itself* is the improvement, not the model.

### Slide 14 — Errors
- Three recurring confusions: `advcl` boundaries, coordination attachment, `nmod` vs. `obl`.
- Driven by Uzbek's flexible word order; future work = contextual embeddings.

---

## 5. Q&A preparation

Likely questions:
1. *"Why a second treebank instead of extending Uzbek-UT?"* — different sources, independent licensing, broader domain (literature + fairy tales + educational), and an opportunity to apply stricter double-annotation with adjudication.
2. *"How were Uzbek-specific UD decisions made?"* — internal guidelines doc, anchored on UD v2; case suffixes as features, postpositions as separate tokens, null copula.
3. *"Did you measure tokenizer/segmentation error?"* — used gold tokenization for parser comparison; future work will couple to a learned tokenizer.
4. *"Why is UDPipe so far behind?"* — older architecture, no contextual representations. Improvement margin shows it is data-bound, not model-bound.
5. *"Cross-lingual transfer from Turkish / Kazakh?"* — flagged in future work.
6. *"Release plan?"* — public release of CoNLL-U + guidelines.

---

## 6. Build instructions

```powershell
cd paper_materials/new-paper/Presentation_Gold_Standard_treebank
latexmk -pdf presentation.tex
```

`latexmkrc` is already present. The `ldvtheme/` folder ships with the template.
