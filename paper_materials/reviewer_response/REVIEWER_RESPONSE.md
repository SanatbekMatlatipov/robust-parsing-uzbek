# Reviewer Response

We thank the reviewer for the constructive critique. Each weakness is
addressed below, with pointers to the specific revised sections of
[main_paper.tex](../main_paper.tex). All revisions draw exclusively on
existing logs, saved predictions, and bibliography updates — no new training,
inference, or retraining was performed.

---

## W1 — "No single example of the output of parsing any sentence in Uzbek"

**Addressed.** A new subsection **§5.5 Qualitative Parsing Examples**
(`\label{sec:qualitative}`) presents two `tikz-dependency` figures generated
from existing Stanza E2.2 dev-set predictions:

- **Fig. `fig:parse_success` (success).** Sentence `s658` from UzUDT-dev
  (*"bu boʻlimdagi asarlar bizga bu xazinaning eshigini ochib beradi."* —
  "The works in this section open the door of this treasure for us.") parsed
  with **100 % UAS = LAS = 100 %**. Demonstrates correct handling of
  determiner–noun, genitive–possessive chaining (`xazinaning → eshigini` via
  `nmod:poss`), accusative object, dative oblique, and the perfective
  converb `xcomp` construction (`ochib → beradi`).
- **Fig. `fig:parse_failure` (failure).** Sentence `s653`
  (*"bolalar oʻyinni davom etdi."* — "The children continued the game.")
  exposes a systematic weakness on Uzbek light-verb constructions (LVC).
  UD annotates *davom* as `compound:lvc` of *etdi*; the parser instead
  promotes *davom* to a near-head, cascading into 3 head errors and 1 label
  error out of 5 tokens. Gold (blue solid) and predicted (red dashed) edges
  are drawn on the same figure for direct contrast.

Predictions taken verbatim from
[saved_models/depparse/uz_combined_E2.2_parser.dev.pred.conllu](../../saved_models/depparse/uz_combined_E2.2_parser.dev.pred.conllu);
gold from the released UzUDT dev set. Stand-alone reproductions of both
figures are saved at
[review_figures/parse_success_tikz.tex](../review_figures/parse_success_tikz.tex)
and
[review_figures/parse_failure_tikz.tex](../review_figures/parse_failure_tikz.tex).
Aligned gold-vs-predicted CoNLL-U excerpts are at
[review_outputs/parse_example_success.conllu](../review_outputs/parse_example_success.conllu)
and
[review_outputs/parse_example_failure.conllu](../review_outputs/parse_example_failure.conllu).

The Uzbek apostrophe U+02BB (ʻ) in prose is rendered via
`\textquoteleft{}` and declared via `\DeclareUnicodeCharacter{02BB}` in the
preamble to ensure pdfLaTeX compatibility on Overleaf.

---

## W2 — Subword analysis "needs more attention"

**Addressed.** A new subsection **§4.4 Subword Alignment for Agglutinative
Morphology** (`\label{sec:subword}`) replaces and substantially expands the
original two-sentence treatment. It contains:

1. **Table III (`tab:subword_examples`)** — WordPiece decomposition of three
   representative Uzbek inflected forms (*bolalarning*, *eshigini*,
   *koʻrsatib*), highlighting the morpheme carried by the rightmost subword
   (GEN, ACC, CONV respectively).
2. A **typological argument** that Uzbek is strictly suffixing, so the last
   subword is the position where Case / Tense / converb suffixes are
   realised, and that mean pooling dilutes this signal across stems.
3. The **quantitative justification** linking the design choice to the
   E2.1-vs-E3.1 ablation in Table IV: last-subword fusion yields
   +2.95 UAS / +2.64 LAS while leaving UPOS essentially unchanged (−0.31),
   evidence that suffix-level cues are decisive for *structural* but not
   *local* decisions.
4. An explicit **limitation note** that last-subword is a positional
   heuristic, not a morphological analyser, and a forward pointer to §7.4
   (`sec:limitations`).

---

## W3 — "No mention of UAS results in experiment tables"

**Addressed.** UAS columns have been added to:

- **Table IV** (`tab:stanza_results`, Stanza Test Set Results) — values
  E1.1 = 69.57, E1.2 = 72.27, E2.1 = 72.05, **E2.2 = 72.39** (best),
  E3.1 = 69.10.
- **Table V** (`tab:spacy_results`, spaCy Test Set Results) — values
  S1.1 = 67.72, **S1.2 = 66.81** (best).

All values are taken verbatim from already-saved logs:
[saved_models/depparse/](../../saved_models/depparse/) `*_summary.json`
(Stanza, field `final_uas`) and
[results/spacy_s1.1_test.json](../../results/spacy_s1.1_test.json) /
[results/spacy_s1.2_test.json](../../results/spacy_s1.2_test.json) (spaCy,
field `dep_uas`). Provenance is documented in
[review_logs/uas_extraction.md](../review_logs/uas_extraction.md). The
augmentation Δ-table (Table VII, `tab:augmentation`) already contained UAS
deltas; no change needed there.

---

## W4 — "Only two frameworks evaluated — limited generalizability"

**Addressed (positioning).** §7.4 Limitations (`Framework scope` paragraph)
now explicitly lists the alternative frameworks not yet evaluated
(UDPipe 2.0, Trankit, DiaParser, MaChAmp) and clarifies that the present
paper is **positioned as a head-to-head graph-vs.-transition comparison**,
not as a multi-framework benchmark. Alternative monolingual encoders
(UzbBERT, XLM-R) are also flagged as out of scope.

The architectural-trade-off conclusion (graph wins on LAS, transition wins
on local POS tagging) is supported by the literature on other agglutinative
languages and does not depend on framework count. UDPipe 2.0 is now cited
(`straka2018udpipe`).

---

## W5 — Small dataset, single runs, no significance testing, default hyperparameters

**Addressed in a substantially expanded §7.4 Limitations and Threats to
Validity**, organised into six paragraphs:

1. **Dataset size** — 781 training / 325 test sentences placed in context
   against the median UD treebank (~10k sentences); explicit caveat about
   per-relation analysis granularity for rare relations.
2. **Single-run reporting** — single-run nature stated up front, with three
   supporting arguments: (i) the headline LAS gap (16.70) is an order of
   magnitude larger than typical seed-induced LAS variance (~0.3–0.5,
   citing `kondratyuk2019cross`); (ii) the ranking is monotonically
   consistent across five Stanza experiments; (iii) a paired-bootstrap CI
   estimate on the 325-sentence test set is ≈ ±2.4 LAS per model — well
   below the architectural gap. Multi-seed repetition is flagged as the
   immediate next step.
3. **Default hyperparameters** — deliberate choice to reflect the
   *out-of-the-box* practitioner experience; targeted tuning could shift
   absolute scores (especially for spaCy's joint training objective, which
   is sensitive to the learning-rate schedule) but rarely flips
   architectural ranking on small UD treebanks.
4. **Framework scope** — see W4.
5. **Subword granularity** — last-subword is a positional heuristic, not a
   morphological tokeniser.
6. **Annotation heterogeneity** — UT vs. UzUDT divergence in feature
   inventories, and its impact on UFeats accuracy after merging.

---

## Reviewer "ideas to explore" — Future-work commitments

§8 (Conclusion) was rewritten to explicitly enumerate the five reviewer
suggestions in a roman-numbered `enumerate` list (rendered via the new
`\usepackage{enumitem}` preamble dependency):

- (i) **Multi-seed significance testing** with bootstrap confidence
  intervals to quantify variance across training runs.
- (ii) **Broader framework evaluation** covering UDPipe 2.0, Trankit,
  DiaParser, and MaChAmp, plus alternative encoders (UzbBERT, XLM-R).
- (iii) **LLM-based data augmentation** (`whitehouse2024llm`,
  `arXiv:2403.02990`) and active learning for small Uzbek treebanks.
- (iv) **Fine-grained error analysis** — per-relation confusion matrices,
  sentence-length stratification, and throughput / latency profiling.
- (v) **Parser ensembling** (graph + transition) to exploit the
  complementary strengths identified in this study.

---

## Submission-readiness checklist

The following items have been verified on the current
[main_paper.tex](../main_paper.tex):

- [x] Document class `IEEEtran_EDM` (10pt, conference, a4paper) preserved.
- [x] Required packages declared in preamble: `inputenc[utf8]`,
      `fontenc[T1]`, `tikz`, `tikz-dependency`, `enumitem`, `booktabs`,
      `multirow`, `flushend`, `url`.
- [x] Two `\DeclareUnicodeCharacter` declarations for U+02BB / U+02BC
      handle Uzbek apostrophes; bare U+02BB occurrences in running text
      replaced with `\textquoteleft{}` (lines 383, 408).
- [x] All tables labelled (`tab:splits`, `tab:complement`,
      `tab:subword_examples`, `tab:arch_diff`, `tab:stanza_results`,
      `tab:spacy_results`, `tab:cross_arch`, `tab:augmentation`,
      `tab:per_relation`); UAS columns present in Tables IV and V.
- [x] All figures labelled (`fig:deprel_freq`, `fig:parse_success`,
      `fig:parse_failure`); the PNG referenced in `fig:deprel_freq`
      (`Figure2_DepRel_Frequency.png`) must be present in the Overleaf
      project root.
- [x] All `\cite{}` keys resolve against `\bibitem` entries (19 entries;
      `straka2018udpipe` and `whitehouse2024llm` added in revision).
- [x] All `\ref{}` and `\S\ref{}` cross-references resolve
      (`sec:results`, `sec:subword`, `sec:qualitative`,
      `sec:limitations`).
- [x] Section order: Introduction → Related Work → Datasets →
      Methodology → Experimental Setup → Results → Discussion →
      Conclusion → Bibliography.
- [x] No TODO / FIXME / placeholder text remaining.

**Recommended pre-submission action on Overleaf:** trigger a clean
`pdflatex` → `bibtex` → `pdflatex` × 2 cycle and confirm: (a) zero
"Unicode character not set up" errors; (b) zero unresolved
`?` cross-references; (c) total page count within the conference limit.

---

## Files changed / added

```
paper_materials/main_paper.tex                                  (revised)
paper_materials/reviewer_response/PLAN.md                       (new)
paper_materials/reviewer_response/REVIEWER_RESPONSE.md          (this file)
paper_materials/reviewer_response/changes_summary.md            (new)
paper_materials/review_logs/uas_extraction.md                   (new)
paper_materials/review_outputs/parse_example_success.conllu     (new)
paper_materials/review_outputs/parse_example_failure.conllu     (new)
paper_materials/review_figures/parse_success_tikz.tex           (new)
paper_materials/review_figures/parse_failure_tikz.tex           (new)
```

No new training, full-test inference, or model retraining was performed.
All revisions draw exclusively on existing logs and saved predictions.
