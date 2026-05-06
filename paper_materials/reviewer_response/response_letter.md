# Response to Reviewers — *Annotated Universal Dependencies Dataset for Literary and Educational Uzbek Texts*

**Manuscript:** UzUDT — Data in Brief submission
**Authors:** Sanatbek Matlatipov, Mersaid Aripov, Makhmud Bobokandov, Gayrat Matlatipov

We thank the reviewer for the careful and constructive comments. Below we
respond to each point individually. Page/section references refer to the
revised manuscript (`UzUDT_DiB_Revised_Clean.md`); a summary of the
corresponding edits is provided in the change log at the end of this letter.

> *Items marked* **[AUTHOR INPUT NEEDED]** *require the authors to confirm /
> supply a value before the letter is sent to the editor.*

---

## 1. Generalisation to noisier domains (social media, technical text)

> *Reviewer:* "Authors should elaborate on how well a model trained on UzUDT
> might generalize to more 'noisy' domains like social media or technical
> documentation."

**Response.** We agree. UzUDT was deliberately constructed from edited
literary and educational text and is therefore a *gold-standard, low-noise*
benchmark; we do not claim it is a representative sample of all Uzbek text.
We have expanded the **Limitations** section of the revised manuscript to
discuss this directly. Specifically, we now explain that:

* Uzbek social-media text exhibits substantial code-switching with Russian,
  mixed Latin/Cyrillic script, non-standard transliteration variants
  (e.g., absence of the diacritic on `oʻ`/`gʻ`), informal contractions,
  emojis, and elliptical/non-sentential utterances — none of which are
  present in UzUDT.
* Technical documentation contributes named entities, transliterated English
  loanwords, formulaic register, and SYM-class tokens (mathematical
  operators, units, currency signs) — and SYM does not occur in UzUDT.
* A model trained solely on UzUDT can therefore be expected to generalise
  reasonably well to other edited Uzbek prose (literary, pedagogical,
  encyclopedic) but its accuracy on social-media text in particular will
  drop until UzUDT is supplemented with in-domain annotated data. We have
  identified this domain extension as a priority for future releases.

The revised passage is in §**Limitations** of the manuscript.

---

## 2. Concrete example of how dependency relations help sentiment analysis

> *Reviewer:* "Could authors provide a specific example or hypothetical use
> case of how dependency relations directly improve the accuracy of
> sentiment models compared to simple bag-of-words approaches?"

**Response.** We address this in the response letter rather than the
manuscript itself, in keeping with the *Data in Brief* convention of
keeping the article focused on the dataset. A concrete illustrative
example follows.

Consider the Uzbek restaurant review:

> *taomlari mazali emas, lekin xizmati a'lo darajada.*
> ("The food is not tasty, but the service is excellent.")

A bag-of-words sentiment classifier is faced with the conflicting cues
`mazali` (positive: *tasty*), `emas` (negation: *not*), `a'lo` (positive:
*excellent*), and the discourse marker `lekin` (*but*). Without syntactic
structure it cannot determine *which* aspect (`taomlari` "food" vs.
`xizmati` "service") each evaluative term modifies, nor which clause is
negated.

With UD-style dependency annotations the structure becomes explicit:

* `mazali` ←`amod`/`xcomp`— `taomlari`, with the auxiliary negator `emas`
  attached as `aux` to the predicate of the *first* clause → the FOOD
  aspect carries **negative** polarity.
* `a'lo` ←`amod`— `darajada` ←`obl`— predicate of the *second* clause,
  with subject `xizmati` (`nsubj`) → the SERVICE aspect carries
  **positive** polarity.

For aspect-based sentiment analysis (ABSA), where the goal is to attach a
polarity label to each (aspect, target) pair, the `nsubj`, `obj`, `amod`,
`obl`, and `advcl` arcs encoded by UzUDT directly identify the target of
each evaluative phrase and which clause the negator scopes over —
information that is simply unrecoverable from a bag-of-words representation.
This is precisely the use case motivating the link to our prior ABSA work
on Uzbek (refs [10] and [11] in the manuscript).

---

## 3. Baseline benchmark scores

> *Reviewer:* "Authors should indicate a baseline experiment that offers a
> benchmark score for the community."

**Response.** We have added a new **Baseline Benchmark** section to the
manuscript reporting test-set scores for two widely used neural NLP
toolkits — Stanza and spaCy — trained and evaluated on UzUDT only. The
configurations are: (i) Stanza + FastText `cc.uz.300`; (ii) Stanza +
TahrirchiBERT (last-subword fusion); (iii) spaCy + FastText. All scores
use the official CoNLL-2018 evaluation protocol (UPOS, XPOS, UFeats, UAS,
LAS).

The numbers reported in **Table 7** of the revised manuscript are taken
verbatim from the JSON summary files released alongside the treebank
(`saved_models/depparse/uz_uzudt_E1.1_parser_summary.json`,
`saved_models/depparse/uz_uzudt_E2.1_parser_summary.json`,
`results/spacy_s1.1_test.json`). A more extensive set of experiments
(cross-treebank augmentation, alternative pooling strategies) is reported
in a companion modelling paper to keep the *Data in Brief* article
focused on the dataset itself.

---

## 4. IAA on the full 681-sentence corpus

> *Reviewer:* "The manuscript notes high agreement (0.90–0.95) during the
> calibration phase. Was the IAA measured again for the full 681-sentence
> corpus before adjudication to assess the difficulty of the entire
> dataset?"

**Response.** This is a fair question and we have clarified the wording
in the revised manuscript. To summarise:

* The IAA values **0.95 (lemma) / 0.95 (UPOS) / 0.90 (morphological
  feature–value bundles)** were measured on the **calibration
  subset only**, before full-corpus annotation began. We have made this
  explicit in step 3 of the **Annotation Workflow**.
* For the full 681-sentence corpus we used a *double-annotation +
  adjudication* procedure rather than computing a separate post-hoc
  agreement statistic. Every sentence was independently annotated by two
  annotators; INCEpTION's comparison view surfaced all token-level
  disagreements, which were resolved in regular adjudication meetings,
  with a senior linguist as final arbiter. The released treebank therefore
  reflects a **consensus gold standard**, and any residual annotator
  disagreement on the full corpus has been resolved during adjudication —
  it is not retained in the released files.
* We did not compute a separate "raw" Cohen's κ / Krippendorff's α on the
  full corpus prior to adjudication because the pre-adjudication
  double-annotation files were treated as intermediate working artefacts
  and were not retained alongside the gold release. We agree this would
  have been a useful diagnostic, and we will retain pre-adjudication files
  for future releases so that a full-corpus IAA can be reported.

The wording in §**Annotation Workflow** has been revised accordingly.

---

## 5. Nature of the warnings recorded in `eval.log`

> *Reviewer:* "Authors stated that the dataset passed validation with
> 'warnings recorded in eval.log.' Could authors specify the nature of
> these warnings? Are they related to rare Uzbek-specific structures or
> technical formatting issues?"

**Response.** We have expanded the **Technical Validation and
Partitioning** subsection to enumerate the warning categories. The
released treebank passes `validate.py` at level 2 with **no errors**; the
recorded warnings fall into three groups:

1. **Low-frequency tag/feature warnings.** The UD tool warns when a
   dependency relation, UPOS tag, or feature–value pair occurs only a
   handful of times in the corpus. Because UzUDT contains 7,542 tokens,
   several legitimate but rare phenomena (e.g., `compound:redup`,
   `flat:foreign`, certain `Voice` and `Evident` values) fall below the
   tool's default frequency threshold. These warnings reflect *corpus
   size*, not annotation inconsistency.
2. **Uzbek-specific morphosyntactic patterns.** A few warnings flag
   constructions where the tool's heuristics expect Indo-European-style
   features — for example, finite-verb chains and null-copula clauses
   typical of Turkic agglutinative morphology. Each was reviewed by the
   linguistic team and confirmed to be guideline-conformant.
3. **UD documentation / metadata warnings.** Finally, the tool emits
   warnings related to the README/treebank metadata (genre tagging,
   contributor list) and a low *split score* triggered because each split
   contains fewer than the 10,000-token threshold the UD tool prefers for
   a fully credited three-way split. These are documentation-side
   warnings, not annotation issues.

None of the warnings indicate annotation errors and all are reproducible
by re-running `validate.py` on the released files.

---

## 6. Per-domain sentence counts in train/test splits

> *Reviewer:* "Could authors provide the exact count of sentences from
> each domain (literary vs. educational vs. fairy tales) in both the
> training and test sets to ensure perfect replicability?"

**Response.** We agree this information is essential for replicability
and have added **Table 6** to the manuscript with per-domain sentence
counts. The split was stratified at the sentence level so that each of
the three source domains appears in both train and test in approximately
its corpus-level proportion.

> **[AUTHOR INPUT NEEDED]** Please supply the exact per-domain counts
> from the source-tracking metadata used during annotation (Maqar /
> Kun shundan boshlanadi / educational texts / ertak.uz fairy tales).
> The placeholders in **Table 6** of the revised manuscript should be
> replaced with these numbers before final submission. Row totals
> (Train = 483, Test = 198, Total = 681) are already populated from the
> released CoNLL-U files.

---

## 7. Copyright / permission for "Maqar" (2023) and "Kun shundan boshlanadi" (2020)

> *Reviewer:* "Authors must confirm that the use of these texts for a
> public dataset complies with Uzbek copyright law or that explicit
> permission was obtained from the publishers."

**Response.** We acknowledge the importance of this point. The released
treebank does **not** redistribute the source texts in continuous form;
it contains only individual annotated sentences (CoNLL-U records),
together with their UD-layer annotations, and is shared under
CC BY-SA 4.0 as required by the Universal Dependencies project. Such use
of short, non-contiguous excerpts for academic linguistic annotation is
generally accepted under fair-use / academic citation provisions in Uzbek
copyright law (see Article 27 of the Law of the Republic of Uzbekistan
"On Copyright and Related Rights"), and analogous practice is standard
across UD treebanks built from copyrighted literary sources (e.g., several
Turkish and Russian UD treebanks).

> **[AUTHOR INPUT NEEDED]** Please confirm one of the following options
> so we can finalise this paragraph for the published response:
>
> 1. *Permission obtained.* Author S. Matkarim is the author of both
>    works and explicit written permission to use sentences from "Maqar"
>    (2023) and "Kun shundan boshlanadi" (G'afur G'ulom Publishing House,
>    2020) for academic UD annotation has been granted. *(If so, please
>    attach the permission letter / email.)*
> 2. *Fair-use position.* No explicit permission was sought; the use is
>    relied on under fair-use / academic citation provisions of Uzbek
>    copyright law because (a) only short, non-contiguous sentences are
>    redistributed, (b) the purpose is non-commercial linguistic research,
>    and (c) full bibliographic attribution is provided in the release.
> 3. *Other arrangement.* (Please describe.)

The corresponding fairy-tale material from `ertak.uz` is in the public
domain or released under a permissive license; the educational material
is from publicly distributed pedagogical resources.

---

## 8. Sections that underwent the most significant AI-assisted revision

> *Reviewer:* "Authors stated that Gemini Pro was used to improve
> 'language, readability, and academic coherence.' Could the authors
> specify which sections underwent the most significant AI-assisted
> revision?"

**Response.** AI-assisted (Gemini Pro) editing was used **only for
language polishing** — grammar, register, and sentence-level clarity —
and never for generating annotation decisions, statistical figures, or
factual claims. All quantitative content (corpus counts, IAA values,
benchmark scores) was produced by the authors from the source data and
verified by re-running the relevant scripts.

> **[AUTHOR INPUT NEEDED]** Please confirm the following list of sections
> that received the most AI-assisted polishing, or amend it as needed:
>
> * Abstract
> * Background
> * Value of the Data (bullet list)
> * Limitations
>
> The Specifications Table, Annotation Workflow, statistical tables, and
> Ethics / CRediT / Acknowledgements sections were authored and edited
> directly by the authors with no AI-assisted rewriting beyond minor
> grammatical correction.

---

## 9. Table 1 — CoNLL-U sample is hard to read

> *Reviewer:* "In the 'Sample of the annotated CoNLL-U format' (Table 1),
> some entries appear truncated or merged or unclear, and it is difficult
> to place any text in a specific column."

**Response.** The original Table 1 was rendered as plain space-separated
text, which collapsed when typeset. In the revised manuscript we present
the same example in two complementary forms:

1. A properly formatted **table** with one column per CoNLL-U field
   (ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC), so
   that the reader can read each annotation field directly.
2. A **monospaced code block** containing the raw, tab-separated CoNLL-U
   record (with `# sent_id` and `# text` comment lines), so that the
   example can be copy-pasted into UD tooling without reformatting.

Both renderings appear in §**Data Description** of the revised manuscript.

---

## 10. Table 3 — only morphological features are listed

> *Reviewer:* "The text mentions 'Table 3' provides a breakdown of POS,
> morphological features, and dependencies. However, the provided Table 3
> only lists morphological features and values."

**Response.** The reviewer is correct — Table 3 in the original submission
covered only morphological feature–value pairs, while the prose referred
to all three layers. We have split the original Table 3 into three
separate, clearly labelled tables in the revised manuscript:

* **Table 3** — UPOS tag distribution (16 of 17 categories; SYM absent).
* **Table 4** — top morphological feature–value pairs (the original
  content of Table 3).
* **Table 5** — top 20 dependency relations (out of 38 in total).

The full distributions remain available in the released `stats.xml` for
readers who require the canonical figures.

---

## 11. Inconsistent terminology: "morphological feature bundles" vs. "feature–value pairs"

> *Reviewer:* "Authors refer to 'morphological feature bundles' and
> 'feature-value pairs' interchangeably. For clarity, authors should
> stick to one term throughout the 'Methodology' section."

**Response.** We thank the reviewer for catching this. We have
standardised the terminology throughout the manuscript as follows:

* **feature–value pair** — a single annotation such as `Case=Nom` or
  `Number=Plur`. Used for individual annotations and in Table 4.
* **feature–value bundle** — the complete set of feature–value pairs
  attached to a single token (e.g., `Case=Nom|Number=Plur` is one
  bundle). Used only in the IAA discussion to make clear that
  agreement was measured on full bundles, not on individual features.

A clarifying sentence defining both terms has been added at the end of
§**Data Description**, and all other occurrences in §**Experimental
Design, Materials and Methods** have been adjusted to use the correct
term.

---

## Summary of changes to the manuscript

| # | Reviewer point | Section affected | Type of change |
|---|----------------|------------------|----------------|
| 1 | Noisy-domain generalisation | Limitations | Expanded |
| 2 | Sentiment-analysis example | (response letter only) | — |
| 3 | Baseline benchmark | New **Baseline Benchmark** section + Table 7 | New |
| 4 | Full-corpus IAA wording | Annotation Workflow, step 3 | Clarified |
| 5 | Nature of `eval.log` warnings | Technical Validation and Partitioning | Expanded |
| 6 | Per-domain split counts | Technical Validation and Partitioning + Table 6 | New |
| 7 | Copyright / permission | (response letter only — pending author confirmation) | — |
| 8 | AI-assisted sections | (response letter only — pending author confirmation) | — |
| 9 | Table 1 readability | Data Description (Table 1 + code block) | Reformatted |
| 10 | Table 3 layer coverage | Tables 3, 4, 5 | Split into three |
| 11 | Terminology consistency | Data Description + Annotation Workflow | Standardised |

We hope these revisions adequately address the reviewer's concerns and
thank the reviewer once more for the constructive feedback.

Sincerely,
The authors
