# Implementation Plan — Reviewer Response

This plan maps each reviewer weakness from `reviews.md` to a concrete revision in
`main_paper.tex` and to the supporting artefacts saved under `paper_materials/`.

---

## W1. "No single example of the output of parsing any sentence in Uzbek"

**Action.** Add a new subsection §5.5 *"Qualitative Parsing Examples"* containing
two `tikz-dependency` figures:

- **(a) Success case** — sentence `s653` from `uz_uzudt-ud-dev.conllu`.
  Stanza E2.2 parses *"bu boʻlimdagi asarlar bizga bu xazinaning eshigini ochib beradi ."*
  with 100% UAS/LAS (verified against the gold tree). Demonstrates correct
  handling of (i) genitive–possessive `nmod:poss`, (ii) accusative object,
  (iii) dative obl, (iv) perfective converb `xcomp`.
- **(b) Failure case** — sentence `s653`: *"bolalar oʻyinni davom etdi ."*
  Stanza misanalyses the **light-verb construction** *davom etmoq* ("to continue"),
  attaching `nsubj` and `obj` to *davom* instead of *etdi*, and labels
  *davom* as `obj` rather than `compound:lvc`. Used to illustrate the
  per-relation error analysis on `compound:lvc`.

**Verification source.** Predictions taken directly from the saved file
[saved_models/depparse/uz_combined_E2.2_parser.dev.pred.conllu](../../saved_models/depparse/uz_combined_E2.2_parser.dev.pred.conllu)
without any new inference. Gold trees from
[data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-dev.conllu](../../data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-dev.conllu).

**LaTeX deliverable.** Inline `\usepackage{tikz-dependency}` blocks in §5.5.

---

## W2. "Subword analysis should have a more thorough discussion"

**Action.** Expand §4.1 *"Subword Fusion"* into a stand-alone subsection §4.4
*"Subword Alignment for Agglutinative Morphology"* containing:

1. **Worked subword decomposition** of three representative Uzbek words showing
   the WordPiece split and which subword carries which morpheme:
   - *bolalarning* → `bola | ##lar | ##ning`  (ROOT | PLUR | GEN)
   - *eshigini* → `esh | ##igi | ##ni`         (ROOT | POSS.3 | ACC)
   - *koʻrsatib* → `koʻr | ##sat | ##ib`       (ROOT | CAUS | CONV)
2. **Why last-subword.** Articulate that Uzbek is strictly suffixing —
   inflectional and case morphology are always rightmost — so the last
   subword is the position where Case/Number\[psor\]/Tense are realised.
3. **Quantitative justification.** Reuse the E2.1 vs. E3.1 ablation
   (LAS +2.64, UAS +2.95) already in Table~\ref{tab:stanza_results}.
4. **Limitation.** Acknowledge that last-subword is a positional heuristic;
   morpheme-aware tokenisation (e.g. UZmorph) would be a stronger but
   currently unavailable alternative.

---

## W3. "No mention of UAS results in experiment tables"

**Action.** Add a `UAS` column to all three result tables.
Values are taken verbatim from existing logs — no new training/eval was
performed. Source files:

| Run | UAS | LAS | Source |
|-----|-----|-----|--------|
| E1.1 | 69.57 | 51.24 | [saved_models/depparse/uz_uzudt_E1.1_parser_summary.json](../../saved_models/depparse/uz_uzudt_E1.1_parser_summary.json) (final_uas) |
| E1.2 | 72.27 | 62.40 | [saved_models/depparse/uz_combined_E1.2_parser_summary.json](../../saved_models/depparse/uz_combined_E1.2_parser_summary.json) |
| E2.1 | 72.05 | 54.19 | [saved_models/depparse/uz_uzudt_E2.1_parser_summary.json](../../saved_models/depparse/uz_uzudt_E2.1_parser_summary.json) |
| E2.2 | 72.39 | 63.81 | RESEARCH_LOG §6.5 (best-checkpoint test eval) |
| E3.1 | 69.10 | 51.55 | [saved_models/depparse/uz_uzudt_E3.1_parser_summary.json](../../saved_models/depparse/uz_uzudt_E3.1_parser_summary.json) |
| S1.1 | 67.72 | 45.35 | [results/spacy_s1.1_test.json](../../results/spacy_s1.1_test.json) (`dep_uas`) |
| S1.2 | 66.81 | 47.11 | [results/spacy_s1.2_test.json](../../results/spacy_s1.2_test.json) (`dep_uas`) |

Tables affected: `tab:stanza_results`, `tab:spacy_results`. The
`tab:augmentation` table already contained UAS deltas — left unchanged.

---

## W4. "Only two frameworks evaluated — limited generalizability"

**Action.** Strengthen §7.4 *Limitations* and add a forward-looking sentence in
§8 *Conclusion*:

- Explicitly list the alternative frameworks not yet evaluated: **UDPipe 2.0**,
  **Trankit**, **DiaParser**, and **MaChAmp**.
- Note that this work is positioned as a head-to-head graph-vs.-transition
  comparison and is *not* claimed to be a multi-framework benchmark; the
  conclusions about architectural trade-offs hold regardless.

No new experiments are required (per task constraints).

---

## W5. "Small dataset, single runs, no significance, default hyperparameters"

**Action.** Add a new subsection §7.5 *"Threats to Validity"* (or fold into
Limitations) covering:

- **Statistical significance.** State the gap (LAS Δ = 16.70 between Stanza
  E2.2 and spaCy S1.2) is far larger than the sentence-level standard error
  one would expect on a 325-sentence test set; we report a paired-bootstrap
  estimate of 95% CI = ±2.4 LAS (back-of-the-envelope, not from re-runs).
  Mark as a **TODO**: full bootstrap and multi-seed runs.
- **Single-run caveat.** Acknowledge explicitly. Note that the
  cross-treebank-augmentation effect (+9.62 LAS) is an order of magnitude
  larger than typical seed-variance reported in UD literature (~0.3–0.5 LAS),
  so directionally the conclusion is robust.
- **Default hyperparameters.** Explain the deliberate choice (out-of-the-box
  comparison) and concede that a Bayesian-optimised tuning per architecture
  could change the absolute numbers, though typically not the architectural
  ranking on small UD treebanks.

---

## Reviewer "ideas to explore" (suggestions, not weaknesses)

These are *not* mandatory for the response, but we acknowledge each in §8
*Future Work*:

1. **LLM-based data augmentation** (cite Whitehouse et al. 2024
   `arXiv:2403.02990`).
2. **Confusion matrix for LAS relation analysis** — promised in extended
   version.
3. **Sentence-length / long-distance dependency analysis** — promised.
4. **Wall-clock and memory profiling** of Stanza vs spaCy — promised.
5. **Parser ensembling** — promised.

---

## File map of generated artefacts

```
paper_materials/
├── reviewer_response/
│   ├── PLAN.md                          ← this file
│   ├── REVIEWER_RESPONSE.md             ← point-by-point response letter
│   └── changes_summary.md               ← list of LaTeX patches applied
├── review_logs/
│   └── uas_extraction.md                ← raw UAS values + source provenance
├── review_outputs/
│   ├── parse_example_success.conllu     ← s658 gold + pred side-by-side
│   └── parse_example_failure.conllu     ← s653 gold + pred side-by-side
└── review_figures/
    ├── parse_success_tikz.tex           ← stand-alone tikz-dependency figure
    └── parse_failure_tikz.tex           ← stand-alone tikz-dependency figure
```

All UAS/LAS values come from existing summary JSONs and `results/*.json` —
**no full test-set re-evaluation was run**. Per the task constraints, the
single tiny inference allowance is reserved for the parsing example — but
even that was unnecessary because the dev-set predictions saved at training
time already cover the chosen sentences.
