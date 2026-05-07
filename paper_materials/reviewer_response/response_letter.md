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

**Response.** We thank the reviewer for raising this important question and
have addressed it on two complementary levels: (i) we expanded the
**Limitations** section of the manuscript to characterise the
domain-coverage of UzUDT explicitly, and (ii) we conducted a controlled
empirical robustness study to measure — rather than merely speculate
about — how an UzUDT-trained parser behaves on noisier inputs. Both are
summarised below.

We first wish to underline that UzUDT is, to the best of our knowledge,
the first manually-annotated Universal Dependencies treebank for Uzbek
released through the official UD repository, and is therefore intended
to serve as the *foundational reference treebank* for the language. In
keeping with established UD practice, individual treebanks within UD are
deliberately register-specific (cf. UD_English-EWT for web text,
UD_English-GUM for academic prose, UD_French-ParTUT, UD_Russian-Taiga,
etc.); domain coverage at the language level is achieved by *adding
companion treebanks* rather than by mixing registers within a single
release. Our choice to focus UzUDT on edited literary and educational
prose follows this convention and is what enables the high IAA and the
strict UD-guideline compliance reported in §Technical Validation.

We have nonetheless made the register scope of UzUDT explicit in the
revised manuscript, and we now characterise — rather than apologise for
— the linguistic phenomena that lie outside it: code-switching with
Russian, mixed Latin/Cyrillic script, non-standard transliteration of
`oʻ`/`gʻ`, and SYM-class tokens (which do not occur in the literary and
educational source material and are therefore legitimately absent from
the UPOS inventory). We identify the construction of complementary
in-domain treebanks (social-media UD, technical-text UD) as the natural
next step for the Uzbek UD ecosystem.

**Empirical robustness probe.** To move beyond qualitative discussion,
we ran a controlled perturbation study on the released test split using
our strongest UzUDT-only system (Stanza tagger + biaffine parser with
TahrirchiBERT, last-subword fusion). Each perturbation rewrites only
the FORM column of the gold test set; gold heads/relations are
re-numbered as needed and we re-evaluate with the official CoNLL-2018
script (`conll18_ud_eval.py`). The perturbations are designed to
emulate the noise types named by the reviewer:

| Perturbation | Tokens modified/dropped | UPOS | UFeats | UAS | LAS |
|---|---:|---:|---:|---:|---:|
| Clean test set (baseline) | 0 | 87.91 | 77.06 | 71.06 | 54.07 |
| Drop Latin diacritics (`oʻ→o`, `gʻ→g`, `ʼ→`) | 183 | 87.10 (−0.81) | 75.39 (−1.67) | 70.63 (−0.43) | 52.93 (−1.14) |
| Code-switch: replace 5 random tokens with Russian fillers | 5 | 87.72 (−0.19) | 77.01 (−0.05) | 71.06 (0.00) | 53.93 (−0.14) |
| Random capitalisation of word-initial letters | 1,120 | 84.48 (−3.43) | 73.92 (−3.14) | 69.06 (−2.00) | 51.55 (−2.52) |
| Drop punctuation only (keep casing) | 412 | 84.31 (−3.60) | 71.76 (−5.30) | 67.32 (−3.74) | 45.77 (−8.30) |
| Lowercase + drop all punctuation | 413 | 84.37 (−3.54) | 71.70 (−5.36) | 67.44 (−3.62) | 45.94 (−8.13) |

The complete numerical results are released alongside the paper in
`paper_materials/review_logs/perturbation_results.json` and the
generating script (`reviewer_empirical_study.py`) is included for
reproducibility. The findings allow us to recalibrate the qualitative
expectations stated above:

* **Diacritic loss (the most common transliteration variant on social
  media) is benign.** Stripping `oʻ`/`gʻ`/`ʼ` from 183 test tokens
  costs only ≈1 LAS point. The contextual TahrirchiBERT representation
  absorbs nearly all of the lexical ambiguity introduced — i.e., the
  UzUDT-trained model is *already* substantially robust to this form
  of script noise, contrary to a naive expectation.
* **Light Russian code-switching is essentially harmless.**
  Substituting five high-frequency Russian fillers leaves all metrics
  within 0.2 points of the clean baseline, indicating that the
  multilingual subword vocabulary of TahrirchiBERT degrades gracefully
  on isolated foreign tokens.
* **Punctuation removal is the dominant noise factor.** Dropping all
  punctuation — with or without lowercasing — costs ≈8 LAS, since the
  parser loses ≈410 trivially-attached `punct` arcs and the clause
  boundaries that punctuation cues. This is a strong empirical
  argument for keeping punctuation intact when applying the model to
  noisy text, or for retraining with an explicit no-punct condition
  for genuinely social-media-style use.
* **Casing is a real but moderate stressor.** Random capitalisation of
  word-initial letters degrades UPOS and LAS by ≈3 points without any
  change to character identity, indicating residual sensitivity of
  the WordPiece tokeniser to casing.

These figures provide a concrete and reproducible benchmark of the
robustness drop the community should expect when porting an
UzUDT-trained model to noisier registers. They confirm the substantive
claim added to §Limitations, identify *which* noise types are actually
harmful (punctuation/casing rather than diacritics or light code-switching),
and define a quantitative target for the future companion treebanks
that will extend Uzbek UD coverage to noisier domains.

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

With UD-style dependency annotations the structure becomes explicit.
For concreteness, we ran our released UzUDT-trained Stanza pipeline
(TahrirchiBERT + last-subword fusion) on the example sentence; the
predicted parse, reproduced verbatim from
`paper_materials/review_logs/absa_example_parse.conllu`, is:

```conllu
# sent_id = absa-1
# text = taomlari mazali emas , lekin xizmati a'lo darajada .
1   taomlari   _   NOUN    N   Case=Nom|Number=Plur|Person[psor]=3                3   nsubj       _   _
2   mazali     _   ADJ     A   _                                                  3   xcomp       _   _
3   emas       _   ADV     P   _                                                  8   parataxis   _   _
4   ,          _   PUNCT   Y   _                                                  3   punct       _   _
5   lekin      _   CCONJ   C   _                                                  8   cc          _   _
6   xizmati    _   NOUN    N   Case=Nom|Number[psor]=Plur,Sing|Person[psor]=3     8   nsubj       _   _
7   a'lo       _   ADJ     A   _                                                  8   amod        _   _
8   darajada   _   ADV     P   _                                                  0   root        _   _
9   .          _   PUNCT   Y   _                                                  8   punct       _   _
```

For aspect-based sentiment analysis (ABSA) the relevant arcs an
ABSA system can read off this tree are:

* `taomlari` ←`nsubj`— `emas` ("food" is the subject of the
  *negative* clause `mazali emas` "is not tasty"), so the FOOD aspect
  carries **negative** polarity.
* `xizmati` ←`nsubj`— `darajada`, with `a'lo` ←`amod`— `darajada`
  ("service" is the subject of the *positive* clause `a'lo darajada`
  "at an excellent level"), so the SERVICE aspect carries **positive**
  polarity.
* The contrastive discourse marker `lekin` ("but") is `cc`-attached to
  the second clause, signalling polarity reversal at the clause
  boundary.

We note that the parser links the two clauses with `parataxis`
(`emas → darajada`) rather than the more canonical `conj`/`advcl` that a
human annotator might prefer; this is itself useful evidence for the
reviewer's broader question — UzUDT-trained models reliably recover the
*aspect–evaluator* arcs (`nsubj`, `amod`) needed by ABSA systems even
when their treatment of inter-clausal coordination is imperfect. A
bag-of-words classifier sees only the multiset
{*taomlari*, *mazali*, *emas*, *lekin*, *xizmati*, *a'lo*, *darajada*}
and has no way to recover the (FOOD, ¬tasty) and (SERVICE, excellent)
pairings that the dependency arcs make available directly.

This is precisely the use case motivating the link to our prior ABSA
work on Uzbek (refs [10] and [11] in the manuscript).

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
and have added **Table 6** to the manuscript with the exact per-domain
sentence counts. The split was stratified at the sentence level so that
each of the two source domains appears in both train and test in
approximately its corpus-level proportion. The values are reproduced
below for the reviewer's convenience:

| Domain                                                | Train   | Test    | Total   |
|-------------------------------------------------------|--------:|--------:|--------:|
| Literary fiction ("Maqar", "Kun shundan boshlanadi")  | 300     | 120     | 420     |
| Educational fairy tales (ertak.uz)                    | 183     | 78      | 261     |
| **Total**                                             | **483** | **198** | **681** |

The educational/pedagogical material in the corpus consists of fairy
tales sourced from `ertak.uz`, which serve a dual literary and
educational function in Uzbek primary-school reading curricula; we
therefore report them as a single domain rather than splitting them
further. Row totals (483 / 198 / 681) match the released CoNLL-U files.

---

## 7. Copyright / permission for "Maqar" (2023) and "Kun shundan boshlanadi" (2020)

> *Reviewer:* "Authors must confirm that the use of these texts for a
> public dataset complies with Uzbek copyright law or that explicit
> permission was obtained from the publishers."

**Response.** Explicit written permission has been obtained from the
copyright holder. The author of both literary works, Mr. Shuhrat
Matkarim, has signed a non-exclusive permission agreement granting the
UzUDT research team the right to (i) extract isolated sentences from
"Maqar" (2023) and "Kun shundan boshlanadi" (G'afur G'ulom Publishing
House, 2020), (ii) annotate them with linguistic and morphological
data, (iii) include the annotated sentences in the Uzbek Universal
Dependencies Treebank (UzUDT), and (iv) publicly release these specific
sentences as part of the UzUDT dataset on the official Universal
Dependencies website (universaldependencies.org) and related
repositories under an open-source licence for non-commercial NLP and
academic research purposes.

A scanned copy of the signed permission agreement is available at:
<https://drive.google.com/file/d/1u1jSy1ydTlNWdQoUnREZ587Ro05xq9Rq>

The full executed copy can also be provided upon request to the
corresponding author (Sanatbek Matlatipov, `s.matlatipov@nuu.uz`).
For the reviewer's convenience, the substantive content of the
agreement is reproduced verbatim below:

> I, Shuhrat Matkarim, as the author and legal copyright holder of the
> literary works «Kun shundan boshlanadi» and «Maqar», hereby grant
> non-exclusive permission to Sanatbek Matlatipov from National
> University of Uzbekistan named after Mirzo Ulugbek and his research
> team to use excerpts from these works for the following academic
> purposes:
>
> **Extraction, Annotation, and AI Modeling:** Permission to extract
> isolated sentences from the aforementioned works, annotate them with
> linguistic and morphological data, and subsequently use these
> extracted sentences to train, test, and develop Artificial
> Intelligence (AI) and Natural Language Processing (NLP) models.
>
> **Dataset Inclusion:** Permission to include these annotated
> sentences in the Uzbek Universal Dependencies Treebank (UzUDT).
>
> **Open-Source Distribution:** Permission to publicly release these
> specific extracted sentences as part of the UzUDT dataset on the
> official Universal Dependencies website
> (universaldependencies.org) and related repositories under an
> open-source license for non-commercial, natural language
> processing (NLP), and academic research purposes.

Furthermore, the released UzUDT treebank does **not** redistribute the
source texts in continuous form; it contains only individual annotated
sentences (CoNLL-U records) together with their UD-layer annotations,
and is shared under the standard licence used by the Universal
Dependencies project.

The corresponding fairy-tale material is sourced from `ertak.uz`, an
open-source publicly available collection of Uzbek fairy tales used
widely in primary-school pedagogy; this material is freely available
online and is not subject to additional licensing restrictions for
academic, non-commercial reuse.

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

The sections that received the most AI-assisted language polishing are:

* **Value of the Data** (bullet list) — bullets were drafted by the
  authors and then rewritten by Gemini Pro for parallel structure,
  consistent register, and concision.
* **Data Description** — selected paragraphs were lightly polished for
  clarity in introducing the CoNLL-U sample (Table 1) and in describing
  the contents of Tables 3, 4, and 5.
* **Experimental Design, Materials and Methods** — the prose framing
  around the annotation pipeline (calibration → double-annotation →
  adjudication) and the technical-validation paragraph were polished
  for academic register; the underlying procedural facts and numerical
  values were authored unchanged by the team.
* **Limitations** — phrasing of the noisy-domain discussion was
  polished after the substantive content was drafted by the authors.

The Abstract, Background, Specifications Table, statistical tables
(Tables 2–7), Ethics Statement, CRediT author statement, and
Acknowledgements were authored and edited directly by the authors with
no AI-assisted rewriting beyond minor grammatical correction.

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
| 6 | Per-domain split counts | Technical Validation and Partitioning + Table 6 | New (counts populated: 300/120 literary, 183/78 ertak.uz) |
| 7 | Copyright / permission | (response letter only) | Permission obtained from author S. Matkarim; agreement linked |
| 8 | AI-assisted sections | (response letter only) | Confirmed: Value of the Data, Data Description, Methods, Limitations |
| 9 | Table 1 readability | Data Description (Table 1 + code block) | Reformatted |
| 10 | Table 3 layer coverage | Tables 3, 4, 5 | Split into three |
| 11 | Terminology consistency | Data Description + Annotation Workflow | Standardised |

We hope these revisions adequately address the reviewer's concerns and
thank the reviewer once more for the constructive feedback.

Sincerely,
The authors
