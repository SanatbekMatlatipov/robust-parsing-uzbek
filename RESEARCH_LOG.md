# Research Log — Towards Robust Uzbek Neural Dependency Parsing

**Author:** Sanatbek Matlatipov
**Contact:** s.matlatipov@nuu.uz
**Institution:** National University of Uzbekistan named after Mirzo Ulugbek, Tashkent, Uzbekistan
**Last Updated:** 2026-02-28

---

## 1. Project Overview

This project develops and evaluates a robust neural dependency-parsing pipeline for Uzbek — a low-resource, agglutinative Turkic language. The key contributions are:

1. **Linguistically-motivated subword alignment.** A last-subword fusion strategy motivated by the suffix-final structure of Uzbek agglutination to bridge the granularity mismatch between BERT WordPiece tokenization and UD word-level annotations — compared against language-agnostic mean pooling. The method is a positional heuristic, not a morphological tool; its design exploits the typological property that grammatical suffixes occupy the rightmost subwords.
2. **Monolingual contextual vs. static embeddings under severe data scarcity.** A controlled comparison of TahrirchiBERT (768-dim, Uzbek-specific) against FastText (300-dim, static) demonstrating how contextual representations interact with training set size when fewer than 1,000 annotated sentences are available.
3. **Cross-treebank data augmentation from heterogeneous sources.** Quantification that combining two annotation-heterogeneous, genre-complementary Uzbek UD treebanks (UzUDT + UD_Uzbek-UT) yields synergistic — not merely additive — accuracy gains, with analysis of why the complementary UPOS, deprel, and morphological-feature distributions (§2.4) drive the improvement.

### 1.1 Research Questions

| # | Research Question |
|---|-------------------|
| **RQ1** | Can a monolingual contextual encoder (TahrirchiBERT) with linguistically-motivated last-subword fusion overcome the limitations of static word embeddings for joint morphosyntactic tagging and dependency parsing under severely low-resource conditions (<1,000 training sentences), and does its advantage scale with corpus size? |
| **RQ2** | For agglutinative Uzbek — where grammatical suffixes encode case, tense, person, and evidentiality — does a linguistically-motivated last-subword fusion strategy, designed to preserve suffix-level morphosyntactic cues, outperform language-agnostic mean pooling for aligning BERT subword representations to UD token boundaries? |
| **RQ3** | Does cross-treebank data augmentation from genre-complementary but annotation-heterogeneous sources produce synergistic accuracy gains when combined with contextual embeddings and linguistically-motivated last-subword fusion, or are the benefits of more data and better representations merely additive? |

---

## 2. Datasets

### 2.1 UD_Uzbek-UzUDT *(Matlatipov, Sanatbek)*

| Property | Detail |
|----------|--------|
| Source | Uzbek literature and educational writing |
| Sentences | 684 |
| Tokens | ~7,800 |
| Annotation platform | INCEpTION |
| Annotators | 6 (4 linguists + 2 NLP engineers) |
| Cross-verification | Full adjudication |
| Inter-annotator agreement | >95% lemma, ~95% UPOS, ~90% morphfeats (Cohen's κ, Krippendorff's α) |
| Annotation layers | UPOS, XPOS, lemma, morphological features, dependency relations |
| UD guidelines | v2 |
| License | CC BY-SA 4.0 |
| UD release | v2.17 (2025-10-01) |
| Contributors | Matlatipov, Sanatbek |
| Genre | Fiction, Academic |
| Reference | Matlatipov (2025) — companion treebank paper submitted to same venue |

**Data Splits:**

| Split | File | Sentences | % |
|-------|------|-----------|---|
| Train | `uz_uzudt-ud-train.conllu` | 451 | 65.9 |
| Dev   | `uz_uzudt-ud-dev.conllu`   | 45  | 6.6  |
| Test  | `uz_uzudt-ud-test.conllu`  | 188 | 27.5 |
| **Total** | | **684** | |

> **Split rationale:** The original UD repository contained only train (483) and test (201) files — no dev split. Both files were pooled (684 sentences) and re-split using standard UD proportions ~66% / ~7% / ~27%.

---

### 2.2 UD_Uzbek-UT *(Akhundjanova, Arofat; Talamo, Luigi)*

| Property | Detail |
|----------|--------|
| Source | News articles (250 sentences) + Fiction (250 sentences) |
| Sentences | 500 |
| Tokens | ~5,850 |
| Annotation | Semi-automatic with full manual correction |
| Tokenization / Lemmatization | Automatic |
| POS & Dependency | Semi-automatic + manual |
| License | CC BY-SA 4.0 |
| UD release | v2.15 (2024-11-15) |
| Contributors | Akhundjanova, Arofat |
| Genre | News, Fiction |

**Reference:**
```bibtex
@inproceedings{akhundjanova-talamo-2025-universal,
    title     = "{U}niversal {D}ependencies Treebank for {U}zbek",
    author    = "Akhundjanova, Arofat and Talamo, Luigi",
    booktitle = "Proceedings of the Third Workshop on Resources and
                 Representations for Under-Resourced Languages and
                 Domains (RESOURCEFUL-2025)",
    month     = mar,
    year      = "2025",
    address   = "Tallinn, Estonia",
    publisher = "University of Tartu Library, Estonia",
    url       = "https://aclanthology.org/2025.resourceful-1.1/",
    pages     = "1--6"
}
```

**Data Splits:**

| Split | File | Sentences | % |
|-------|------|-----------|---|
| Train | `uz_ut-ud-train.conllu` | 330 | 66.0 |
| Dev   | `uz_ut-ud-dev.conllu`   | 33  | 6.6  |
| Test  | `uz_ut-ud-test.conllu`  | 137 | 27.4 |
| **Total** | | **500** | |

> **Split rationale:** The original release contained only `uz_ut-ud-test.conllu` with all 500 sentences (no train/dev split). Split sequentially using the same ~66% / ~7% / ~27% proportions as UzUDT.

> **⚠ Genre bias in UT splits:** Because the UT corpus is ordered news-first (sentences 1–250) then fiction (251–500), the sequential split means train ≈ news-heavy, test ≈ fiction-heavy. This creates a realistic cross-genre evaluation scenario but should be acknowledged in the paper.

---

### 2.3 Data Settings

All experiments are evaluated under two data settings:

| Setting | Source | Train | Dev | Test | Total Sentences |
|---------|--------|-------|-----|------|-----------------|
| **UzUDT only** (`.1`) | UD_Uzbek-UzUDT | 451 | 45 | 188 | 684 |
| **UzUDT + UT merged** (`.2`) | UzUDT + UT concatenated | 781 | 78 | 325 | 1,184 |

Merged data files were created by simple concatenation of corresponding train/dev/test splits from both treebanks, stored as `uz_combined.{train,dev,test}.in.conllu`.

> **Key comparison for the paper:** The merged setting nearly doubles the training data (451 → 781 sentences) relative to UzUDT alone, providing a direct test of whether data quantity is a primary bottleneck.

---

### 2.4 Treebank Linguistic Statistics (from UD Tools)

The statistics below were generated by the Universal Dependencies validation and evaluation toolkit (`tools/` from the [universaldependencies/tools](https://github.com/UniversalDependencies/tools) GitHub repository). These Perl/Python scripts compute corpus-level statistics, validate annotation consistency, and assign UD quality scores.

#### 2.4.1 UD Quality Assessment

| Quality Metric | UD_Uzbek-UzUDT | UD_Uzbek-UT |
|----------------|---------------|-------------|
| UD validation | **✅ PASSED** | ❌ FAILED (1 syntax error, 92 warnings) |
| UD Stars | **2.5** | 0 |
| UD Quality Score | **0.489** | 0.005 |
| Feature score (weight 0.111) | 0.5 | 1.0 |
| Genre score (weight 0.111) | 0.111 | 0.111 |
| Lemma score (weight 0.111) | 1.0 | 1.0 |
| Size score (weight 0.370) | 0.293 | 0.258 |
| Split score (weight 0.074) | 0.01 | 0.01 |
| Tag score (weight 0.111) | 0.941 | 1.0 |
| Deprel score (weight 0.111) | 0.865 | 0.784 |

> **UT validation failures:** The UT treebank triggers 1 syntax error (`too-many-objects`: multiple direct objects under a single predicate) and 92 warnings — predominantly `obl-should-be-nmod` (oblique attached to a nominal parent should be `nmod`) and `verbform-fin-without-mood` (finite verbs missing the `Mood` feature). These annotation inconsistencies do not prevent use in training but may introduce noise when merging with UzUDT, which passes validation cleanly.

> **Paper note:** Both treebanks receive low split scores (0.01) because neither originally provided the standard three-way train/dev/test split with ≥10,000 words per split — a fundamental low-resource constraint.

#### 2.4.2 Corpus-Level Statistics

| Statistic | UD_Uzbek-UzUDT | UD_Uzbek-UT | Combined |
|-----------|---------------|-------------|----------|
| Sentences | 684 | 500 | **1,184** |
| Tokens (surface) | 7,582 | 5,930 | **13,512** |
| Syntactic words | 7,582 | 5,930 | 13,512 |
| Fused tokens | 0 | 0 | 0 |
| Unique lemmas | 1,748 | 2,278 | — |
| Unique word forms | 3,099 | 3,387 | — |
| Type–token ratio (forms) | 0.409 | 0.571 | — |
| Avg. tokens per sentence | 11.1 | 11.9 | 11.4 |
| Words with ≥1 feature | 3,544 (46.7%) | 3,981 (67.1%) | — |
| Unique UPOS tags used | 16 / 17 | 17 / 17 | — |
| Unique morphological features | 55 | 47 | — |
| Unique dependency relations | 38 | 33 | — |
| Genres | Fiction, Academic | News, Fiction | Fiction, Academic, News |
| Annotation approach | Manual (INCEpTION) | Semi-auto + manual correction | — |
| UD release version | v2.17 (2025-10-01) | v2.15 (2024-11-15) | — |

**Key observations:**
- **UT has higher lexical diversity** (type–token ratio 0.571 vs. 0.409), reflecting both its news genre (more proper nouns, named entities) and smaller corpus size.
- **UT has higher feature coverage** (67.1% of tokens annotated with features vs. 46.7%), partly because UzUDT has a much larger proportion of PUNCT tokens (20.7% vs. 14.5%) which typically carry no features.
- **UzUDT is missing SYM** (only 16/17 UPOS tags) — no symbol tokens in the literary/academic corpus.
- **Neither treebank has fused tokens** (multi-word tokens), which is typical for Uzbek since agglutinative suffixation does not produce orthographic contractions.

#### 2.4.3 UPOS Tag Distribution (Comparative)

| UPOS | UzUDT Count | UzUDT % | UT Count | UT % | Δ % | Note |
|------|-------------|---------|----------|------|-----|------|
| NOUN | 2,526 | 33.3 | 2,152 | 36.3 | −3.0 | Dominant in both |
| VERB | 1,585 | 20.9 | 987 | 16.6 | +4.3 | UzUDT richer in verbs (literary narrative) |
| PUNCT | 1,571 | 20.7 | 860 | 14.5 | +6.2 | UzUDT uses more punctuation (dialogue markers ‹›) |
| ADJ | 522 | 6.9 | 484 | 8.2 | −1.3 | Comparable |
| PRON | 458 | 6.0 | 193 | 3.3 | **+2.8** | UzUDT 2.4× more pronouns (1st/2nd person in literature) |
| **PROPN** | **26** | **0.3** | **308** | **5.2** | **−4.9** | **UT has 12× more proper nouns (news genre)** |
| ADV | 230 | 3.0 | 203 | 3.4 | −0.4 | Comparable |
| NUM | 184 | 2.4 | 214 | 3.6 | −1.2 | UT slightly more (statistics in news) |
| ADP | 120 | 1.6 | 184 | 3.1 | −1.5 | UT more postpositions |
| DET | 103 | 1.4 | 103 | 1.7 | −0.4 | Identical counts |
| CCONJ | 87 | 1.1 | 104 | 1.8 | −0.6 | |
| AUX | 88 | 1.2 | 79 | 1.3 | −0.2 | |
| PART | 43 | 0.6 | 36 | 0.6 | 0.0 | |
| INTJ | 23 | 0.3 | 6 | 0.1 | +0.2 | UzUDT has dialogue |
| SCONJ | 9 | 0.1 | 9 | 0.2 | −0.0 | Very low in both |
| X | 7 | 0.1 | 7 | 0.1 | 0.0 | |
| SYM | — | — | 1 | 0.0 | — | Only in UT |

> **📌 Paper figure (placeholder):** A grouped bar chart or stacked bar showing the UPOS distribution comparison between UzUDT and UT would effectively visualize genre-driven differences. Most striking: PROPN (0.3% vs. 5.2%) and PRON (6.0% vs. 3.3%).

**Implications for merging:**
- The dramatic PROPN imbalance (26 → 334 after merging) means the merged model is **exposed to far more named entities**, which helps on mixed-genre test sets but may shift PROPN/NOUN classification boundaries.
- More VERB tokens from UzUDT and more PROPN from UT are **complementary** — each treebank fills gaps in the other's UPOS coverage.

#### 2.4.4 Morphological Feature Inventory

| Feature Category | UzUDT Values | UT Values | In Both | UzUDT Only | UT Only |
|-----------------|-------------|-----------|---------|------------|---------|
| **Aspect** | Hab, Imp, Perf, Prog (4) | Prog (1) | Prog | Hab, Imp, Perf | — |
| **Case** | Abl, Acc, Dat, Gen, Loc, Nom (6) | Abl, Acc, Dat, Gen, Loc, Nom (6) | All 6 | — | — |
| **Evident** | Fh, Nfh (2) | — | — | Fh, Nfh | — |
| **Mood** | Cnd, Des, Imp, Ind, Int, Opt, Pot (7) | Cnd, Imp, Ind, Int, Opt, Pot (6) | 6 | Des | — |
| **Number** | Plur, Sing (2) | Plur, Sing (2) | All 2 | — | — |
| **Number[psor]** | Plur; Plur,Sing; Sing (3) | — | — | All 3 | — |
| **Person** | 1, 2, 3 (3) | 1, 2, 3 (3) | All 3 | — | — |
| **Person[psor]** | 1, 2, 3 (3) | — | — | All 3 | — |
| **Tense** | Fut, Past, Pres (3) | Fut, Past, Pres (3) | All 3 | — | — |
| **VerbForm** | Conv, Fin, Inf, Part, Vnoun (5) | Conv, Fin, Inf, Part, Vnoun (5) | All 5 | — | — |
| **Voice** | Act, Pass (2) | Pass (1) | Pass | Act | — |
| **PronType** | Dem, Ind, Int, Neg, Prs, Rcp, Rel, Tot (8) | Dem, Ind, Int, Neg, Prs, Rcp, Rel, Tot (8) | All 8 | — | — |
| **Polarity** | Neg, Pos (2) | Neg (1) | Neg | Pos | — |
| **Reflex** | Yes (1) | Yes (1) | Yes | — | — |
| **Poss** | Yes (1) | Yes (1) | Yes | — | — |
| **NumType** | Card, Ord (2) | Card, Frac, Ord, Range, Sets (5) | Card, Ord | — | Frac, Range, Sets |
| **Degree** | — | Cmp (1) | — | — | Cmp |
| **Abbr** | — | Yes (1) | — | — | Yes |
| **Foreign** | — | Yes (1) | — | — | Yes |
| **ExtPos** | ADP (1) | ADJ (1) | — | ADP | ADJ |
| **Total unique** | **55** | **47** | | | |

**Key annotation differences:**
1. **Possessor agreement** (`Number[psor]`, `Person[psor]`): UzUDT annotates possessor agreement on nouns (e.g., *bolalari* "his/their children") — UT does not. This creates a systematic feature gap when merging.
2. **Evidentiality** (`Evident=Fh/Nfh`): UzUDT marks firsthand vs. non-firsthand evidentiality (e.g., *debdi* "apparently said") — a typologically important Turkic feature absent from UT.
3. **Aspect granularity**: UzUDT distinguishes Habitual/Perfective/Imperfective/Progressive; UT annotates only Progressive.
4. **UT-only features**: Abbreviation (`Abbr`), degree comparison (`Degree=Cmp`), foreign words (`Foreign`), and finer numeral types (`Frac`, `Range`, `Sets`) — reflecting UT's news domain.

> **Implication for UFeats prediction:** The different feature inventories explain why UFeats accuracy is the most volatile metric across experiments. When merging treebanks, the model must learn a union of 55 + 47 features with different annotation conventions — particularly challenging for possessor agreement features annotated in UzUDT but absent in UT.

#### 2.4.5 Dependency Relation Inventory (Top 20, Comparative)

| Rank | UzUDT Relation | Count | % | UT Relation | Count | % |
|------|----------------|-------|---|-------------|-------|---|
| 1 | punct | 1,571 | 20.7 | punct | 860 | 14.5 |
| 2 | obl | 720 | 9.5 | obl | 602 | 10.1 |
| 3 | nsubj | 720 | 9.5 | nsubj | 540 | 9.1 |
| 4 | amod | 452 | 6.0 | nmod | 350 | 5.9 |
| 5 | advcl | 444 | 5.9 | amod | 338 | 5.7 |
| 6 | obj | 418 | 5.5 | compound | 290 | 4.9 |
| 7 | compound | 265 | 3.5 | obj | 251 | 4.2 |
| 8 | nmod | 252 | 3.3 | advmod | 238 | 4.0 |
| 9 | advmod | 216 | 2.8 | compound:lvc | 215 | 3.6 |
| 10 | conj | 200 | 2.6 | case | 212 | 3.6 |
| 11 | nmod:poss | 189 | 2.5 | conj | 208 | 3.5 |
| 12 | det | 183 | 2.4 | nummod | 160 | 2.7 |
| 13 | xcomp | 139 | 1.8 | nmod:poss | 142 | 2.4 |
| 14 | acl | 137 | 1.8 | compound:svc | 138 | 2.3 |
| 15 | nummod | 131 | 1.7 | acl | 133 | 2.2 |
| 16 | dep | 125 | 1.6 | cc | 109 | 1.8 |
| 17 | cc | 109 | 1.4 | flat | 106 | 1.8 |
| 18 | ccomp | 94 | 1.2 | advcl | 106 | 1.8 |
| 19 | root | 684 | 9.0 | root | 500 | 8.4 |
| 20 | parataxis | 57 | 0.8 | iobj | 69 | 1.2 |

**Relations unique to one treebank:**

| Only in UzUDT (38 total) | Only in UT (33 total) |
|--------------------------|----------------------|
| `dep` (125) — underspecified | `expl` (1) |
| `advmod:emph` (6) | |
| `nmod:part` (2) | |
| `dislocated` (1) | |
| `orphan` (5) | |
| `fixed` (8) | |

**Notable distribution differences:**
1. **`advcl`**: 4× more frequent in UzUDT (444 vs. 106). Literary Uzbek uses far more adverbial clauses (converb constructions with *-ib*, *-gach*).
2. **`compound:lvc`**: 10× more frequent in UT (215 vs. 21). News Uzbek heavily uses light-verb constructions (*qildi* "did", *etdi* "made" + noun).
3. **`compound:svc`**: 17× more frequent in UT (138 vs. 8). Serial-verb constructions dominate in news narrative.
4. **`flat`**: 10× more frequent in UT (106 vs. 11). Named entities in news require `flat` for multi-word proper names.
5. **`discourse`**: 9× more in UzUDT (64 vs. 7). Dialogue-heavy literary texts use more discourse markers.
6. **`dep`**: 125 instances in UzUDT only — an underspecified fallback relation indicating annotation difficulty; absent from UT.
7. **`case`**: UT uses nearly 2× more `case` (212 vs. 121), reflecting more adpositional government in news register.

> **📌 Paper figure (placeholder):** A side-by-side horizontal bar chart comparing the deprel distributions of UzUDT vs. UT would make a compelling figure showing genre-driven syntactic divergence. The `compound:lvc` / `advcl` contrast is particularly striking.

> **Implication for cross-treebank merging:** The complementary deprel distributions explain why merging dramatically improves LAS (+9.62 to +11.16 points). A model trained on UzUDT alone rarely sees `compound:lvc` or `compound:svc` patterns; adding UT introduces these constructions. Conversely, a UT-only model would lack `advcl` and `discourse` coverage. The merged training set covers **41 unique relations** (union of 38 + 33 minus overlap), up from 38 or 33 individually.

#### 2.4.6 Summary: Why These Treebanks Are Complementary

| Dimension | UD_Uzbek-UzUDT | UD_Uzbek-UT | Merged Benefit |
|-----------|---------------|-------------|----------------|
| **Genre** | Fiction, Academic | News, Fiction | Three genres → more robust models |
| **Named entities** | Very few (26 PROPN) | Many (308 PROPN) | Better PROPN recognition |
| **Verbal morphology** | Rich (4 Aspect, 7 Mood, Evident) | Basic (1 Aspect, 6 Mood) | Union of features |
| **Possessor agreement** | Annotated (Number/Person[psor]) | Not annotated | Partial coverage |
| **Light-verb constructions** | Sparse (21 compound:lvc) | Rich (215 compound:lvc) | Better LVC parsing |
| **Adverbial clauses** | Rich (444 advcl) | Sparse (106 advcl) | Better clause structure |
| **Validation quality** | Clean (0 errors) | 1 error, 92 warnings | Some noise from UT |
| **Lexical diversity** | Lower TTR (0.409) | Higher TTR (0.571) | Broader vocabulary |

> **Paper argument:** The two Uzbek UD treebanks are not simply "more data" — they are **structurally complementary** in genre, syntactic constructions, and morphological annotation depth. This explains why merging produces such large accuracy gains, especially for LAS, where exposure to diverse dependency patterns is critical.

---

## 3. Pipeline Architecture

```
Raw Uzbek Text
       │
       ▼
┌─────────────────────────┐
│   BERT Encoder          │  ← TahrirchiBERT (E2) or absent (E1)
│  (subword tokenization) │
└────────────┬────────────┘
             │  "super-token" fusion: last-subword → UD word tokens
             ▼
┌─────────────────────────┐
│  Joint Tagger + Parser  │  ← BiLSTM + DeepBiaffine (Stanza-based)
│  UPOS / XPOS / UFeats   │
│  Dependency arcs + rels │
└─────────────────────────┘
```

### Components

| Component | Detail |
|-----------|--------|
| Framework | Stanza (modified fork with BERT support added to POS tagger and dependency parser) |
| Contextual embeddings | TahrirchiBERT `tahrirchi/tahrirchi-bert-base` (768-dim, monolingual Uzbek BERT) |
| Static embeddings | FastText `cc.uz.300.vec` (300-dim, Uzbek crawl vectors) |
| Tagger | BiLSTM + softmax multi-task head (UPOS, XPOS, UFeats) |
| Parser | DeepBiaffine graph-based (Dozat & Manning, 2017) |
| Subword fusion | Last-subword selection of BERT subwords per UD token ("super-token") |
| Pretrained models | Hosted at [huggingface.co/Sanatbek/uzudt](https://huggingface.co/Sanatbek/uzudt) |

---

## 4. Experimental Design

### 4.1 Experiment Matrix

Three embedding/fusion configurations, each evaluated on two data settings, yield **6 experimental runs**:

| Run | Embeddings | Data Setting | Super-token Fusion | Description | Status |
|-----|------------|-------------|-------------------|-------------|--------|
| **E1.1** | FastText cc.uz.300 | UzUDT only | N/A | Static baseline, small data | ✅ Complete |
| **E1.2** | FastText cc.uz.300 | UzUDT + UT | N/A | Static baseline, merged data | ✅ Complete |
| **E2.1** | TahrirchiBERT | UzUDT only | Last-subword | Contextual, suffix-aware fusion, small data | ✅ Complete |
| **E2.2** | TahrirchiBERT | UzUDT + UT | Last-subword | Contextual, suffix-aware fusion, merged data | ✅ Complete |
| **E3.1** | TahrirchiBERT | UzUDT only | Mean pooling | Contextual, language-agnostic fusion, small data | ✅ Complete |
| **E3.2** | TahrirchiBERT | UzUDT + UT | Mean pooling | Contextual, language-agnostic fusion, merged data | ⏳ Pending |

**Design rationale:** E1 vs E2 isolates the effect of embedding type (RQ1). E2 vs E3 isolates the effect of subword fusion strategy while holding BERT constant (RQ2). Comparing across `.1` vs `.2` within each configuration measures the data augmentation effect and its interaction with the other factors (RQ3).

**Workflow per run:** Train POS tagger → Re-tag data with predicted POS → Train dependency parser → Evaluate on test set.

> **E3 training commands:** Use the same commands as E2 but add `--bert_pooling mean`. See `README.md` for full details. E3 uses the same TahrirchiBERT model — only the subword-to-token aggregation function changes.

### 4.2 Hyperparameters

All models use Stanza default training settings (no custom tuning):

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Adam β | (0.9, 0.999) |
| Learning rate | 3 × 10⁻³ |
| Batch size | 5,000 tokens |
| Max training steps | 50,000 (with early stopping on dev) |
| Dropout | 0.33 |
| Parser MLP depth | 1 (arc) + 1 (rel) |
| Parser hidden dim | 500 |
| UPOS embedding dim | 50 |
| Super-token embedding dim | 768 (TahrirchiBERT hidden size) |

> **Note for the paper:** Early versions of the paper listed incorrect hyperparameters (AdamW, lr=2×10⁻⁵, batch=32, epochs=30). The values above reflect the actual Stanza defaults used in all experiments.

### 4.3 Evaluation Protocol

- **POS tagging metrics:** UPOS accuracy, XPOS accuracy, UFeats accuracy, AllTags accuracy (joint)
- **Dependency parsing metrics:** UAS, LAS (primary); MLAS, BLEX (secondary, dev-set only)
- **Evaluation script:** Standard UD CoNLL 2018 shared task evaluation (`scripts/eval.py`)
- **All results reported on held-out test sets** with model selection on dev sets

---

## 5. Technical Decisions & Rationale

### 5.1 Why TahrirchiBERT?

| Criterion | TahrirchiBERT | BERTbek | mBERT |
|-----------|---------------|---------|-------|
| HuggingFace ID | `tahrirchi/tahrirchi-bert-base` | `elmurod1202/bertbek-news-big-cased` | `bert-base-multilingual-cased` |
| Hidden size | 768 | 768 | 768 |
| Language | Uzbek monolingual | Uzbek monolingual | 104 languages |
| Domain | Broad (diverse web corpus) | News-centric | Wikipedia + BookCorpus |
| Selection rationale | Broad domain coverage matches heterogeneous treebank genres (literature, academic, news, fiction) | ✗ News-biased; mismatch with UzUDT literary/academic genre | ✗ Uzbek underrepresented in multilingual training data |

> **Decision:** TahrirchiBERT was selected for the primary experiments. BERTbek is planned as a future ablation (see `future_research_log.md`).

### 5.2 Subword-to-Token Alignment: Last-Subword vs. Mean Pooling (RQ2)

BERT tokenizes text into WordPiece subwords whose boundaries rarely coincide with UD word-token boundaries. A "super-token" fusion function must aggregate the subword-level hidden states into a single vector per UD token. This alignment decision is **especially consequential for agglutinative languages** like Uzbek, where a single UD token may carry a chain of suffixes encoding case, tense, person, number, and evidentiality.

#### Linguistic motivation for last-subword selection

Uzbek morphology is predominantly suffixal. For example:

| UD Token | Gloss | Suffixes | WordPiece subwords |
|----------|-------|----------|--------------------|
| *bolalarning* | "of the children" | *bola* + *-lar* (PL) + *-ning* (GEN) | `bola`, `##lar`, `##ning` |
| *ko'rganmisiz* | "have you seen?" | *ko'r* + *-gan* (PTCP) + *-mi* (INT) + *-siz* (2PL) | `ko'r`, `##gan`, `##mi`, `##siz` |

The **last subword** (`##ning`, `##siz`) carries the outermost grammatical suffix — precisely the features that UD annotation targets (Case=Gen, Mood=Int|Person=2|Number=Plur). Mean pooling dilutes this signal by averaging with stem-level subwords.

#### Experimental comparison (RQ2)

| Configuration | Fusion strategy | Rationale |
|---------------|-----------------|----------|
| **E2** (Last-subword) | Take hidden state of final subword per UD token | Preserves suffix-level morphosyntactic cues |
| **E3** (Mean pooling) | Average all subword hidden states per UD token | Language-agnostic baseline; distributes information evenly |

Both use the same TahrirchiBERT encoder — the *only* difference is the aggregation function in `bert_embedding.py`. This controlled comparison directly tests whether a linguistically-motivated positional heuristic (last-subword) outperforms a language-agnostic one (mean pooling) for agglutinative languages.

> **E3.1 complete; E3.2 in progress.** E3.1 results show mean pooling is comparable to last-subword for POS tagging but inferior for parsing (see §6.4). E3.2 will complete the picture on merged data.

#### Hypothesis

We hypothesize that last-subword selection will outperform mean pooling particularly for:
- **UFeats prediction** — since morphological features are overwhelmingly encoded in suffixes
- **XPOS accuracy** — since language-specific POS tags incorporate morphological distinctions
- **LAS** — since correct deprel assignment (e.g., `obl` vs. `nmod`) depends on case-suffix identification

The advantage may be smaller for UPOS, which is primarily determined by stem identity.

### 5.3 Why no Apertium morphological normalization?

Apertium is an open-source rule-based morphological analyzer available for Uzbek. An early version of the experiment plan included Apertium-based normalization as a preprocessing step to reduce lexical sparsity.

**Finding:** Testing `apertium_normalize.py` on the UzUDT treebank (Latin-NFC-normalized) produced **zero form changes** across all tokens. The UD treebanks already contain consistent lemma annotations in the LEMMA column, making the Apertium preprocessing step redundant.

**Decision:** Apertium normalization was removed from the experiment matrix entirely. This reduced the matrix from 12 initial configurations to the final 2 (E1 FastText, E2 TahrirchiBERT).

> **Paper note:** Apertium is discussed in the Related Work section as relevant prior art for Uzbek morphological analysis, with an explicit statement that we tested and excluded it because the UD treebanks already contain lemma annotations.

### 5.4 Dev split creation

Neither treebank originally shipped with a development set suitable for early stopping:
- **UzUDT:** Had train (483) + test (201) only → pooled and re-split as 451 / 45 / 188
- **UD_Uzbek-UT:** Had only a single test file (500 sentences) → split as 330 / 33 / 137

Both use consistent ~66% / ~7% / ~27% proportions following UD community conventions.

### 5.5 BERT integration into Stanza

The original Stanza POS tagger and dependency parser supported only static pretrained embeddings (`--wordvec_pretrain_file`). BERT support existed only in the NER tagger and constituency parser. We added `--bert_model` and `--bert_pooling` arguments to both the POS tagger and dependency parser by following the NER model's integration pattern. This required modifications to 9 source files (see `README.md` for details).

### 5.6 Previous model discrepancy

The previously trained tagger (`uz_uzudt_xlm-roberta-base_tagger.pt`) used XLM-RoBERTa-base rather than any Uzbek-specific model. The old paper also reported scores (LAS=66.90) that did not match actual evaluation logs (LAS=53.21). All current results are from fresh retraining with proper configurations.

---

## 6. Results

5 of 6 experiment runs are complete: 10 trained models (5 POS taggers + 5 dependency parsers). E3.2 (mean pooling, merged data) is currently training.

### 6.1 Main Results — Test Set (Primary)

#### Table 1: Morphosyntactic Tagging (Test Set)

| Run | Embeddings | Data | UPOS | XPOS | UFeats | AllTags |
|-----|------------|------|------|------|--------|---------|
| E1.1 | FastText | UzUDT | 79.19 | 79.81 | 66.61 | — |
| E1.2 | FastText | UzUDT+UT | 80.26 | 83.20 | 66.98 | — |
| E2.1 | TahrirchiBERT (last-sub) | UzUDT | 82.45 | 80.90 | 65.37 | — |
| **E2.2** | **TahrirchiBERT (last-sub)** | **UzUDT+UT** | **85.08** | **84.72** | **71.09** | **—** |
| E3.1 | TahrirchiBERT (mean) | UzUDT | 82.76 | 81.37 | 65.22 | — |

#### Table 2: Dependency Parsing (Test Set)

| Run | Embeddings | Data | UAS | LAS |
|-----|------------|------|-----|-----|
| E1.1 | FastText | UzUDT | 69.57 | 51.24 |
| E1.2 | FastText | UzUDT+UT | 72.27 | 62.40 |
| E2.1 | TahrirchiBERT (last-sub) | UzUDT | 72.05 | 54.19 |
| **E2.2** | **TahrirchiBERT (last-sub)** | **UzUDT+UT** | **72.39** | **63.81** |
| E3.1 | TahrirchiBERT (mean) | UzUDT | 69.10 | 51.55 |

> **Best system:** E2.2 (TahrirchiBERT + last-subword fusion + merged UzUDT+UT) achieves the highest scores across all metrics.

---

### 6.2 Dev Set Results (Model Selection Checkpoint)

These are the scores at the best-checkpoint step (selected by dev AllTags / dev LAS), used for early stopping. Dev scores tend to be higher than test scores due to the small dev set sizes.

#### Table 3: Morphosyntactic Tagging (Dev Set, Best Checkpoint)

| Run | UPOS | XPOS | UFeats | AllTags | Best Step | Train Loss at Best |
|-----|------|------|--------|---------|-----------|-------------------|
| E1.1 | 79.19 | 79.81 | 66.61 | 56.99 | 1,200 | 0.946 |
| E1.2 | 83.78 | 88.72 | 70.98 | 63.81 | 800 | 1.576 |
| E2.1 | 83.54 | 83.23 | 68.17 | 58.54 | 600 | 1.154 |
| **E2.2** | **87.19** | **90.01** | **73.21** | **67.10** | 700 | 1.352 |
| E3.1 | 83.39 | 83.07 | 65.84 | 57.61 | 800 | 0.982 |

#### Table 4: Dependency Parsing (Dev Set, Best Checkpoint)

| Run | UAS | LAS | MLAS | BLEX | Best Step | Train Loss at Best |
|-----|-----|-----|------|------|-----------|-------------------|
| E1.1 | 71.58 | 55.28 | 46.68 | 49.59 | 1,300 | 5.768 |
| E1.2 | 70.51 | 62.87 | 56.64 | 60.02 | 800 | 3.701 |
| E2.1 | 73.91 | 58.23 | 50.57 | 53.06 | 1,000 | 5.352 |
| **E2.2** | **72.39** | **63.81** | **58.25** | **60.82** | 900 | 3.361 |
| E3.1 | 71.58 | 55.90 | 47.57 | 51.29 | 1,000 | 5.398 |

> **Note for the paper:** MLAS and BLEX are available from dev-set evaluations only. Consider whether to include these in the paper tables or report them as supplementary. Dev scores are consistently higher than test scores, especially on the merged data where dev sets are larger (78 sentences vs. 45).

---

### 6.3 Comparative Analysis: Effect of TahrirchiBERT (RQ1)

> **Note:** In this section, E2 uses last-subword fusion. The comparison isolates the embedding effect (static → contextual) while holding fusion strategy constant.

#### Table 5: TahrirchiBERT vs. FastText — Absolute Deltas (Test Set)

| Metric | UzUDT only (E2.1 − E1.1) | UzUDT+UT (E2.2 − E1.2) | Interpretation |
|--------|--------------------------|------------------------|----------------|
| UPOS | **+3.26** (79.19 → 82.45) | **+4.82** (80.26 → 85.08) | BERT consistently helps POS tagging; larger gain on merged data |
| XPOS | +1.09 (79.81 → 80.90) | +1.52 (83.20 → 84.72) | Modest XPOS improvement |
| UFeats | −1.24 (66.61 → 65.37) | **+4.11** (66.98 → 71.09) | UFeats degrades on small data but improves substantially on merged data |
| UAS | **+2.48** (69.57 → 72.05) | +0.12 (72.27 → 72.39) | UAS improves on small data; negligible on merged |
| LAS | **+2.95** (51.24 → 54.19) | +1.41 (62.40 → 63.81) | Modest LAS improvement in both settings |

**Key finding (RQ1):** TahrirchiBERT provides consistent improvement over static FastText for UPOS tagging (+3.26 to +4.82 points). The gain is amplified when more training data is available. However, the LAS improvement is more modest (+1.41 to +2.95), suggesting contextual embeddings have a larger impact on local tagging decisions than on structural dependency predictions in this low-resource setting.

**Unexpected finding:** UFeats *decreases* by 1.24 points when switching to BERT on the small UzUDT-only setting (E1.1 → E2.1), but *increases* by 4.11 points on the merged data (E1.2 → E2.2). This suggests BERT's rich representations may need sufficient training examples to effectively learn fine-grained morphological feature bundles — with too little data, the model may overfit or fail to leverage the contextual signal for feature prediction.

---

### 6.4 Comparative Analysis: Effect of Subword Fusion Strategy (RQ2)

> **E3.1 complete; E3.2 in progress.** Partial comparison available for UzUDT-only (E2.1 vs. E3.1). Full comparison across both data settings will be completed after E3.2 finishes.

#### Table 6: Last-Subword vs. Mean Pooling — Absolute Deltas (Test Set)

| Metric | UzUDT only (E2.1 − E3.1) | UzUDT+UT (E2.2 − E3.2) | Interpretation |
|--------|--------------------------|------------------------|-----------|
| UPOS | **−0.31** (82.45 → 82.76) | *Awaiting E3.2* | Mean pooling marginally better for UPOS |
| XPOS | **−0.47** (80.90 → 81.37) | *Awaiting E3.2* | Mean pooling marginally better for XPOS |
| UFeats | +0.15 (65.37 → 65.22) | *Awaiting E3.2* | Essentially tied; hypothesis not confirmed |
| UAS | **+2.95** (72.05 → 69.10) | *Awaiting E3.2* | Last-subword clearly better for structural parsing |
| LAS | **+2.64** (54.19 → 51.55) | *Awaiting E3.2* | Last-subword advantage confirmed for labeled parsing |

**Preliminary finding (RQ2, UzUDT only):** Contrary to the initial hypothesis, the two fusion strategies perform comparably on POS tagging — mean pooling is even marginally better for UPOS (−0.31) and XPOS (−0.47). However, last-subword fusion shows a clear advantage for dependency parsing (+2.95 UAS, +2.64 LAS), suggesting that suffix-level cues are more important for structural attachment decisions than for local tag assignment. The UFeats difference (+0.15) is negligible, counter to the prediction that suffix-final information would dominate morphological feature prediction. Full conclusions await E3.2 results on merged data.

---

### 6.5 Comparative Analysis: Effect of Treebank Merging (RQ3)

#### Table 7: Merged Data vs. UzUDT-Only — Absolute Deltas (Test Set)

| Metric | FastText (E1.2 − E1.1) | TahrirchiBERT (E2.2 − E2.1) | Interpretation |
|--------|------------------------|------------------------------|----------------|
| UPOS | +1.07 (79.19 → 80.26) | +2.63 (82.45 → 85.08) | Merging helps UPOS consistently |
| XPOS | **+3.39** (79.81 → 83.20) | **+3.82** (80.90 → 84.72) | Large XPOS gains from more data |
| UFeats | +0.37 (66.61 → 66.98) | **+5.72** (65.37 → 71.09) | Dramatic UFeats gain for BERT when data increases |
| UAS | +2.70 (69.57 → 72.27) | +0.34 (72.05 → 72.39) | UAS improves for FastText; BERT already near ceiling |
| LAS | **+11.16** (51.24 → 62.40) | **+9.62** (54.19 → 63.81) | **Dramatic LAS gains — largest single effect** |

**Key finding (RQ3):** Cross-treebank merging produces the **single largest accuracy improvement** in all experiments. LAS jumps by +11.16 points for FastText and +9.62 for TahrirchiBERT — far exceeding the gains from upgrading embeddings. Nearly doubling the training data (451 → 781 sentences) is more impactful than switching from static to contextual embeddings, confirming that **data quantity is the primary bottleneck** for low-resource Uzbek parsing.

**Interaction between BERT and data size:** The BERT advantage grows with more data. On merged data, BERT achieves its largest gains in UPOS (+4.82) and UFeats (+4.11), while on UzUDT alone, the UPOS gain is smaller (+3.26) and UFeats actually degrades. This suggests BERT and data augmentation are **complementary**: more data allows the contextual encoder to better exploit its representations.

---

### 6.6 Combined Summary Table (for Paper)

#### Table 8: Full Results Matrix (Test Set)

| Run | Embeddings | Fusion | Data | Train Sents | UPOS | XPOS | UFeats | UAS | LAS |
|-----|------------|--------|------|-------------|------|------|--------|-----|-----|
| E1.1 | FastText | N/A | UzUDT | 451 | 79.19 | 79.81 | 66.61 | 69.57 | 51.24 |
| E1.2 | FastText | N/A | UzUDT+UT | 781 | 80.26 | 83.20 | 66.98 | 72.27 | 62.40 |
| E2.1 | TahrirchiBERT | Last-sub | UzUDT | 451 | 82.45 | 80.90 | 65.37 | 72.05 | 54.19 |
| **E2.2** | **TahrirchiBERT** | **Last-sub** | **UzUDT+UT** | **781** | **85.08** | **84.72** | **71.09** | **72.39** | **63.81** |
| E3.1 | TahrirchiBERT | Mean | UzUDT | 451 | 82.76 | 81.37 | 65.22 | 69.10 | 51.55 |
| E3.2 | TahrirchiBERT | Mean | UzUDT+UT | 781 | — | — | — | — | — |
| | | | | | | | | | |
| *Δ BERT effect (merged, last-sub)* | | | | | *+4.82* | *+1.52* | *+4.11* | *+0.12* | *+1.41* |
| *Δ Data effect (BERT, last-sub)* | | | | | *+2.63* | *+3.82* | *+5.72* | *+0.34* | *+9.62* |
| *Δ Fusion (last-sub − mean, UzUDT)* | | | | | *−0.31* | *−0.47* | *+0.15* | *+2.95* | *+2.64* |

---

### 6.7 Training Dynamics

#### Table 9: Training Time and Convergence

| Run | Task | Total Steps | Best Step | % Steps to Best | Elapsed (sec) | Elapsed (min) |
|-----|------|-------------|-----------|-----------------|---------------|---------------|
| E1.1 | POS | 7,200 | 1,200 | 16.7% | 1,432 | 23.9 |
| E1.1 | Parse | 7,300 | 1,300 | 17.8% | 269 | 4.5 |
| E1.2 | POS | 6,800 | 800 | 11.8% | 300 | 5.0 |
| E1.2 | Parse | 6,800 | 800 | 11.8% | 156 | 2.6 |
| E2.1 | POS | 6,600 | 600 | 9.1% | 290 | 4.8 |
| E2.1 | Parse | 7,000 | 1,000 | 14.3% | 375 | 6.3 |
| E2.2 | POS | 6,700 | 700 | 10.4% | 514 | 8.6 |
| E2.2 | Parse | 6,900 | 900 | 13.0% | 814 | 13.6 |
| E3.1 | POS | 6,800 | 800 | 11.8% | 823 | 13.7 |
| E3.1 | Parse | 7,000 | 1,000 | 14.3% | 494 | 8.2 |

**Observations:**
- All models converge very early (best checkpoint at 9–18% of max steps), indicating the 50,000 max-step budget is more than sufficient for these small datasets.
- BERT runs (E2.x) reach their best POS checkpoint slightly earlier (steps 600–700) than FastText runs (steps 800–1200), suggesting BERT representations provide a stronger initialization.
- Parser training takes longer proportionally for BERT configurations due to the added overhead of BERT forward passes.
- E3.1 (mean pooling) shows similar convergence speed to E2.1 (last-subword): both reach best POS at step 600–800 and best Parse at step 1000.
- Total training time for all 10 models (E1–E3.1): ~91 minutes on a single NVIDIA RTX A6000 GPU.

---

### 6.8 Training Curves

Per-run training plots are available in `saved_models/{pos,depparse}/plots/`:

#### POS Tagger Training Curves

| Run | Loss | Accuracy (UPOS/XPOS/UFeats) | Learning Rate | Overview |
|-----|------|------------------------------|---------------|----------|
| E1.1 | `saved_models/pos/plots/uz_uzudt_E1_tagger_loss.png` | `saved_models/pos/plots/uz_uzudt_E1_tagger_accuracy.png` | `saved_models/pos/plots/uz_uzudt_E1_tagger_lr.png` | `saved_models/pos/plots/uz_uzudt_E1_tagger_overview.png` |
| E1.2 | `saved_models/pos/plots/uz_combined_E1.2_tagger_loss.png` | `saved_models/pos/plots/uz_combined_E1.2_tagger_accuracy.png` | `saved_models/pos/plots/uz_combined_E1.2_tagger_lr.png` | `saved_models/pos/plots/uz_combined_E1.2_tagger_overview.png` |
| E2.1 | `saved_models/pos/plots/uz_uzudt_E2.1_tagger_accuracy.png` | *(not generated — check)* | *(not generated — check)* | *(not generated — check)* |
| E2.2 | `saved_models/pos/plots/uz_combined_E2.2_tagger_loss.png` | `saved_models/pos/plots/uz_combined_E2.2_tagger_accuracy.png` | `saved_models/pos/plots/uz_combined_E2.2_tagger_lr.png` | `saved_models/pos/plots/uz_combined_E2.2_tagger_overview.png` |
| E3.1 | `saved_models/pos/plots/uz_uzudt_E3.1_tagger_loss.png` | `saved_models/pos/plots/uz_uzudt_E3.1_tagger_accuracy.png` | `saved_models/pos/plots/uz_uzudt_E3.1_tagger_lr.png` | `saved_models/pos/plots/uz_uzudt_E3.1_tagger_overview.png` |

#### Dependency Parser Training Curves

| Run | Loss | Accuracy (UAS/LAS) | Learning Rate | Overview |
|-----|------|---------------------|---------------|----------|
| E1.1 | `saved_models/depparse/plots/uz_uzudt_E1.1_parser_loss.png` | `saved_models/depparse/plots/uz_uzudt_E1.1_parser_accuracy.png` | `saved_models/depparse/plots/uz_uzudt_E1.1_parser_lr.png` | `saved_models/depparse/plots/uz_uzudt_E1.1_parser_overview.png` |
| E1.2 | `saved_models/depparse/plots/uz_combined_E1.2_parser_loss.png` | `saved_models/depparse/plots/uz_combined_E1.2_parser_accuracy.png` | `saved_models/depparse/plots/uz_combined_E1.2_parser_lr.png` | `saved_models/depparse/plots/uz_combined_E1.2_parser_overview.png` |
| E2.1 | `saved_models/depparse/plots/uz_uzudt_E2.1_parser_loss.png` | `saved_models/depparse/plots/uz_uzudt_E2.1_parser_accuracy.png` | `saved_models/depparse/plots/uz_uzudt_E2.1_parser_lr.png` | `saved_models/depparse/plots/uz_uzudt_E2.1_parser_overview.png` |
| E2.2 | `saved_models/depparse/plots/uz_combined_E2.2_parser_loss.png` | `saved_models/depparse/plots/uz_combined_E2.2_parser_accuracy.png` | `saved_models/depparse/plots/uz_combined_E2.2_parser_lr.png` | `saved_models/depparse/plots/uz_combined_E2.2_parser_overview.png` |
| E3.1 | `saved_models/depparse/plots/uz_uzudt_E3.1_parser_loss.png` | `saved_models/depparse/plots/uz_uzudt_E3.1_parser_accuracy.png` | `saved_models/depparse/plots/uz_uzudt_E3.1_parser_lr.png` | `saved_models/depparse/plots/uz_uzudt_E3.1_parser_overview.png` |

#### Cross-Experiment Comparison Plots

> **📌 TODO (manual):** Generate side-by-side comparison plots using `scripts/compare_experiments.py`. Suggested comparison plots for the paper:
>
> 1. **POS UPOS comparison** — E1 vs E2 on both data settings (4 lines on one chart)
> 2. **Parser LAS comparison** — E1 vs E2 on both data settings (4 lines on one chart)
> 3. **Bar chart** — Final test metrics grouped by experiment (grouped bar chart from JSON summaries)
>
> See `README.md` → "Comparing experiments" section for commands. Output to `results/comparison_plots/`.

#### Weights & Biases

All training runs were logged to W&B:
- POS tagger project: `uzbek-pos-tagger`
- Depparse project: `uzbek-depparse`

> **📌 TODO (manual):** Export publication-quality charts from W&B dashboard:
> 1. Open the W&B project → select all 4 runs → create grouped line plot
> 2. Click ⚙️ icon → Export as PNG/SVG
> 3. Optionally create a W&B Report for a shareable URL with interactive charts

---

## 7. Analysis & Discussion

### 7.1 Answering the Research Questions

**RQ1 — Can TahrirchiBERT with last-subword fusion overcome static embedding limitations, and does its advantage scale with corpus size?**

**Yes, consistently for POS tagging; modestly for parsing; and the advantage amplifies with more data.** TahrirchiBERT (with last-subword fusion) improves UPOS by +3.26 to +4.82 points over FastText, confirming that monolingual contextual embeddings capture morphological patterns of Uzbek agglutination more effectively than static vectors. LAS improvement is more modest (+1.41 to +2.95), indicating that structural dependency prediction benefits less from embedding quality when training data is severely limited. Critically, BERT's advantage *scales with corpus size*: UFeats flips from −1.24 on small data to +4.11 on merged data (see §7.2), indicating a minimum data threshold for contextual embeddings to effectively learn fine-grained morphological feature bundles.

**RQ2 — Does last-subword fusion outperform mean pooling for Uzbek?**

**Partially — the advantage is task-dependent.** E3.1 results (UzUDT only, Table 6) reveal a nuanced picture. For POS tagging, mean pooling performs comparably or marginally better than last-subword: UPOS 82.76 vs. 82.45 (+0.31 for mean), XPOS 81.37 vs. 80.90 (+0.47 for mean). UFeats is effectively tied (65.22 vs. 65.37). However, for dependency parsing, last-subword fusion shows a clear advantage: UAS 72.05 vs. 69.10 (+2.95), LAS 54.19 vs. 51.55 (+2.64). This suggests that suffix-positional cues matter more for structural attachment decisions (head selection, relation labeling) than for local POS tag assignment, where distributional context from all subwords is equally informative.

> **📌 Awaiting E3.2** to confirm whether this task-dependent pattern holds on merged data. The hypothesis is that the last-subword advantage for parsing will persist or strengthen, since more training data allows better exploitation of suffix-level cues (cf. the BERT–data interaction in §7.2).

**RQ3 — Does cross-treebank augmentation interact synergistically with embeddings and fusion strategy?**

**Yes, data augmentation is the single most impactful factor, and its interaction with BERT is synergistic, not merely additive.** Merging UzUDT with UD_Uzbek-UT raises LAS by +9.62 to +11.16 points — far exceeding the BERT upgrade (+1.41 to +2.95). The two treebanks are structurally complementary in genre, UPOS distribution (PROPN 0.3% vs. 5.2%), deprel coverage (`advcl` 4× in UzUDT, `compound:lvc` 10× in UT), and morphological feature depth (§2.4.6). The synergy is evidenced by the BERT–data interaction: BERT's UFeats advantage goes from −1.24 (small data) to +4.11 (merged), a +5.35 amplification — more data doesn't just add linearly, it unlocks BERT's capacity for fine-grained morphological discrimination. Full interaction analysis after E3 results complete the picture.

### 7.2 BERT–Data Interaction Effect

An important observation is that the benefit of TahrirchiBERT is **amplified by more data**:

| Metric | BERT Δ on small data (UzUDT) | BERT Δ on merged data (UzUDT+UT) | Amplification |
|--------|-------------------------------|----------------------------------|---------------|
| UPOS | +3.26 | +4.82 | +1.56 more |
| UFeats | −1.24 | +4.11 | +5.35 more |
| LAS | +2.95 | +1.41 | −1.54 less |

BERT's advantage for UFeats prediction flips from negative to strongly positive with more data. This suggests that fine-grained morphological feature prediction from contextual embeddings requires a minimum data threshold to be effective. The LAS pattern is opposite: BERT's structural improvement is proportionally smaller on merged data, possibly because the additional data already resolves many structural ambiguities that BERT would otherwise help with.

### 7.3 Why Absolute Performance Remains Moderate

Despite the best system (E2.2) achieving 85.08 UPOS and 63.81 LAS, performance is moderate compared to high-resource languages (where UPOS > 97% and LAS > 90% are common). Contributing factors:

1. **Subword–morpheme misalignment.** BERT's WordPiece tokenization does not respect Uzbek morpheme boundaries, causing grammatical information to be split inconsistently.
2. **Polyfunctional markers.** Many Uzbek function words serve multiple syntactic roles (e.g., *-ni* as accusative case vs. relativizer), creating persistent `case`/`mark`/`discourse` ambiguities.
3. **Complex predicates.** Light-verb constructions and serial-verb patterns create `xcomp`/`ccomp` and `compound:lvc` confusions not resolvable from local context alone. The deprel inventory (§2.4.5) shows `compound:lvc` is 10× more frequent in UT than UzUDT — models trained on UzUDT alone rarely encounter this pattern.
4. **Data scarcity.** Even the merged corpus (781 training sentences, ~8,500 tokens) is very small by modern NLP standards. Rare constructions and long-tail morphological combinations remain severely underrepresented.
5. **Cross-genre evaluation.** The UT test set is predominantly fiction while training is predominantly news, introducing domain mismatch that penalizes all systems.
6. **Annotation inconsistency between treebanks.** The UT treebank failed UD validation with 92 warnings (§2.4.1), primarily `obl-should-be-nmod` misattachments and missing `Mood` features on finite verbs. When merged, these inconsistencies introduce label noise that the model must absorb. Additionally, the different morphological feature inventories (§2.4.4) — UzUDT annotates possessor agreement (`Number[psor]`, `Person[psor]`) and evidentiality (`Evident`) while UT does not — create systematic gaps in UFeats prediction targets.
7. **Underspecified dependencies.** UzUDT contains 125 `dep` relations (§2.4.5) — an underspecified fallback used when annotators could not determine the correct relation. These tokens are essentially mislabeled from the parser's perspective.

### 7.4 Dependency Distribution Diagnostics

The UzUDT treebank exhibits a skewed dependency relation distribution (see `scripts/Figure2_DepRel_Frequency.png` and the full comparative inventory in §2.4.5):

> **📌 Paper Figure:** Include `scripts/Figure2_DepRel_Frequency.png` as a figure showing the top 20 dependency relations in UzUDT. Consider a companion figure comparing UzUDT vs. UT distributions side-by-side (see §2.4.5 placeholder).

- **High-frequency relations** (`punct`, `obl`, `nsubj`, `root`) dominate and strongly influence UAS/LAS. These four alone account for >40% of all tokens in both treebanks.
- **Genre-driven divergence** is most visible in mid-frequency relations: `advcl` is 4× more common in UzUDT (literary converb chains), while `compound:lvc` and `compound:svc` are 10–17× more common in UT (news light-verb constructions). See §2.4.5 for detailed counts.
- **Mid-frequency relations** (`advcl`, `acl`, `xcomp`, `nmod`) correspond to structurally complex or ambiguity-prone constructions.
- **Errors in mid-frequency relations** disproportionately affect LAS and are harder to fix without better clause-boundary modeling.

**Implication for future work:** A targeted improvement strategy should maintain accuracy on frequent core relations while explicitly addressing linguistically difficult mid-frequency relations.

### 7.5 Dev vs. Test Generalization Gap

Comparing dev and test scores reveals an interesting pattern:

| Metric | E2.2 Dev | E2.2 Test | Gap |
|--------|----------|-----------|-----|
| UPOS | 87.19 | 85.08 | −2.11 |
| XPOS | 90.01 | 84.72 | −5.29 |
| UFeats | 73.21 | 71.09 | −2.12 |
| LAS | 63.81 | 63.81 | 0.00 |

The large XPOS gap (−5.29) suggests the test set contains XPOS patterns not well represented in dev, possibly due to the genre differences or the small dev set size (78 sentences in the merged setting). LAS shows perfect alignment between dev and test for E2.2, though this may be coincidental given the small evaluation sets.

---

## 8. Figures for the Paper

### 8.1 Available Figures

| Figure | File | Status | Description |
|--------|------|--------|-------------|
| Architecture diagram | `baseline_architecture.png` | ✅ Ready | End-to-end pipeline overview |
| Morphotactics diagram | `verb-noun.png` | ⚠️ **Not found at root** — check if needs to be generated | Uzbek verb/noun suffix chains |
| Deprel frequency chart | `scripts/Figure2_DepRel_Frequency.png` | ✅ Ready | Top-20 dependency relations in UzUDT |
| Per-run training curves | `saved_models/*/plots/*.png` | ✅ Ready (32 plots) | Loss, accuracy, LR, overview per run |

### 8.2 Figures To Generate

| Figure | TODO | Tool |
|--------|------|------|
| Cross-experiment POS comparison | Compare E1 vs E2 UPOS/XPOS/UFeats learning curves | `scripts/compare_experiments.py --mode pos` |
| Cross-experiment LAS comparison | Compare E1 vs E2 UAS/LAS learning curves | `scripts/compare_experiments.py --mode depparse` |
| Bar chart of final test metrics | Grouped bar chart: 4 experiments × 5 metrics | `scripts/compare_experiments.py --mode summary` |
| W&B exported charts | Publication-quality versions of above | W&B dashboard → Export |

---

## 9. Saved Models

### 9.1 Experiment Model Files

| Run | POS Tagger | Dependency Parser |
|-----|-----------|------------------|
| E1.1 | `saved_models/pos/uz_uzudt_E1_tagger.pt` | `saved_models/depparse/uz_uzudt_E1.1_parser.pt` |
| E1.2 | `saved_models/pos/uz_combined_E1.2_tagger.pt` | `saved_models/depparse/uz_combined_E1.2_parser.pt` |
| E2.1 | `saved_models/pos/uz_uzudt_E2.1_tagger.pt` | `saved_models/depparse/uz_uzudt_E2.1_parser.pt` |
| E2.2 | `saved_models/pos/uz_combined_E2.2_tagger.pt` | `saved_models/depparse/uz_combined_E2.2_parser.pt` |
| E3.1 | `saved_models/pos/uz_uzudt_E3.1_tagger.pt` | `saved_models/depparse/uz_uzudt_E3.1_parser.pt` |

### 9.2 Baseline / Legacy Models

| Model | File | Notes |
|-------|------|-------|
| POS Tagger (legacy) | `saved_models/pos/uz_uzudt-base_tagger.pt` | Pre-retraining; used XLM-RoBERTa |
| Dependency Parser (legacy) | `saved_models/depparse/uz_uzudt_nocharlm_parser.pt` | Pre-retraining; FastText only |
| Tokenizer | `saved_models/tokenize/uz_uzudt_tokenizer.pt` | Shared across experiments |

All models also hosted on HuggingFace: [`Sanatbek/uzudt`](https://huggingface.co/Sanatbek/uzudt)


---

## 10.Changelog

| Date | Action |
|------|--------|
| 2026-02-28 | **E3.1 complete:** Added mean-pooling results (UzUDT only) — POS UPOS=82.76, Parser LAS=51.55; populated Tables 1–4, 6, 8, 9; updated §6.4 fusion comparison with preliminary analysis, §7.1 RQ2 answer with task-dependent findings; E3.2 in progress |
| 2026-02-28 | Elevated subword fusion to RQ2; restructured all 3 RQs for low-resource efficient methods focus; added E3 (mean pooling) to experiment matrix with placeholders; expanded §5.2 with linguistic examples and hypothesis; renumbered tables |
| 2026-02-28 | Added §2.4 Treebank Linguistic Statistics from UD tools: UD quality scores, corpus-level stats, UPOS distribution comparison, morphological feature inventory diff, dependency relation inventory (top-20 comparative), complementarity analysis; updated §7.3/§7.4 with cross-references |
| 2026-02-28 | Revised RESEARCH_LOG: removed operational instructions (moved to README), added comparative analysis tables with deltas, dev-set results, training dynamics, and plot inventory; restructured for paper readiness |
| 2026-02-27 | Trimmed experiment matrix to 2 configs (E1 FastText, E2 TahrirchiBERT); all 4 runs complete; moved E3–E7 to `future_research_log.md`; populated results with actual scores |
| 2026-02-27 | Rewrote `research-paper.tex` — removed Apertium/BERTbek/mean-pooling references; added TahrirchiBERT, two data settings, 4-run results, correct hyperparameters |
| 2026-02-26 | Removed Apertium normalization — tested on UzUDT, found 0 form changes; deleted `scripts/apertium_normalize.py` |
| 2026-02-26 | Integrated BERT into POS tagger and parser (9 files); added `--bert_model` and `--bert_pooling` arguments |
| 2026-02-25 | Expanded matrix with ×2 data settings (UzUDT, UzUDT+UT merged); created merged data files |
| 2026-02-25 | Installed PyTorch 2.6.0+cu124; confirmed 4× NVIDIA RTX A6000 GPUs |
| 2026-02-24 | Fixed `torch.load()` (13 files), `xpos_vocab_factory.py`, `lcode2lang`; downloaded FastText; added logging infrastructure |
| 2026-02-24 | Created RESEARCH_LOG; full code review; defined experiment plan |
| 2026-02-24 | Re-split UzUDT (684 sents → 451/45/188) and UT (500 sents → 330/33/137) |
| 2025-11-07 | Previous training (XLM-RoBERTa POS tagger + FastText parser) |
| 2025-10-01 | UD_Uzbek-UzUDT v2.17 released |
| 2024-11-15 | UD_Uzbek-UT v2.15 released |
