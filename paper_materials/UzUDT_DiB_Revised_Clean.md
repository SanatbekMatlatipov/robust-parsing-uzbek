# ARTICLE INFORMATION

**Article title**

Annotated Universal Dependencies Dataset for Literary and Educational Uzbek Texts

**Authors**

_Sanatbek Matlatipov<sup>1</sup>, Mersaid Aripov<sup>1</sup>, Makhmud Bobokandov<sup>1</sup> and Gayrat Matlatipov<sup>2</sup>_

**Affiliations**

_<sup>1</sup>: National University of Uzbekistan named after Mirzo Ulugbek,_

_Universitet street, 4, Olmazor district, 100174, Tashkent city, Uzbekistan;_

_<sup>2</sup>:_ Urgench State University named after Abu Rayhan Biruni_,_

Khamid Alimdjan, 14, 220100, Urgench City, Uzbekistan_;_

**Corresponding author’s email address and Twitter handle**

[_s.matlatipov@nuu.uz_](mailto:s.matlatipov@nuu.uz)_;_ [_mr.sanatbek@gmail.com_](mailto:mr.sanatbek@gmail.com)

**Keywords**

Dependency Treebank; Natural Language Processing; Turkic languages; Uzbek language; Syntax; Morphological features; Low-resource languages; Part-of-Speech tagging

**Abstract**

This data article describes an Uzbek Universal Dependencies (UD) treebank released as a manually curated gold-standard dataset. The resource contains 681 sentences (7,542 tokens) drawn from literary and educational Uzbek texts, providing a domain-specific complement to previously available web-based or news-oriented materials \[1\]. Annotation was carried out in the INCEpTION environment \[7\] by a five-member team comprising three linguists and two NLP engineers. The workflow followed the UD v2 framework and included calibration, full-corpus double annotation, and adjudication to improve annotation consistency. Agreement on the shared calibration material was high across lemmatization, universal part-of-speech annotation, and morphological features \[9\]. The released dataset includes lemmas, UPOS tags, morphological features, and basic dependency relations in standard CoNLL-U format, and it has been validated for compatibility with the Universal Dependencies ecosystem. As an openly reusable Uzbek syntactic resource, it can support the development and evaluation of POS taggers, morphological analyzers, and dependency parsers, while also enabling comparative and cross-lingual studies for low-resource languages \[10\].

# SPECIFICATIONS TABLE

<div class="joplin-table-wrapper"><table><tbody><tr><td><p><strong>Subject</strong></p></td><td><p>Computer Sciences</p></td></tr><tr><td><p><strong>Specific subject area</strong></p></td><td><p>Natural Language Processing (NLP); Computational Linguistics; Uzbek language; Dependency Parsing for Low-Resource Agglutinative Languages.</p></td></tr><tr><td><p><strong>Type of data</strong></p></td><td><p>Table (CoNLL-U), Text (Raw sentences), Analyzed, Filtered, Processed</p><p>CoNLL-U Format<strong> </strong>The dataset uses the CoNLL-U format, a standard for Universal Dependencies annotation [6]. Files are UTF-8 plain text where each sentence is separated by a blank line, and comment lines start with a hash (#). Each word token is represented by a single line containing 10 tab-separated fields:</p><ol><li><strong><em>ID</em>:</strong> Word index (integer starting at 1).</li><li><strong><em>FORM</em>:</strong> The word form or punctuation.</li><li><strong><em>LEMMA</em>:</strong> The lemma or stem.</li><li><strong><em>UPOS</em>:</strong> Universal part-of-speech tag.</li><li><strong><em>XPOS</em>:</strong> Language-specific part-of-speech tag (underscore if unavailable).</li><li><strong><em>FEATS</em>:</strong> List of morphological features.</li><li><strong><em>HEAD</em>:</strong> ID of the current word's head (0 if root).</li><li><strong><em>DEPREL</em>:</strong> Universal dependency relation to the Head.</li><li><strong><em>DEPS</em>:</strong> Enhanced dependency graph (optional).</li><li><strong><em>MISC</em>:</strong> Any other annotation (e.g., spacing).</li></ol></td></tr><tr><td><p><strong>Data collection</strong></p></td><td><p>The data was collected by manual annotation of sentences from publicly available Uzbek literature works: the fiction story "Maqar" [2] and the book "Kun shundan boshlanadi" [3], and from educational texts and fairy tales at ertak.uz. The annotation was performed in the INCEpTION platform by a team of linguists and NLP engineers using a gold-standard workflow that included calibration, double annotation, adjudication, and validation. The final dataset was checked with the official UD validation suite (<em>validate.py</em>) to ensure compliance with UD v2 standards. All sentences are in the Uzbek Latin script (the official script for modern Uzbek), and we avoided sentences containing non-standard or heavily dialectal forms. We targeted relatively self-contained sentences of moderate length (the average sentence length is ~12 tokens), avoiding extremely long or overly simple sentences.</p></td></tr><tr><td><p><strong>Data source location</strong></p></td><td><p>Institution: National University of Uzbekistan named after Mirzo Ulugbek</p><p>City: Tashkent</p><p>Country: Uzbekistan</p></td></tr><tr><td><p><strong>Data accessibility</strong></p></td><td><p>Repository name: Universal Dependencies (GitHub/Official Website)</p><p>Data identification number: <strong><em>uzudt</em></strong></p><p>Direct URL to data:<strong> </strong><a href="https://github.com/UniversalDependencies/UD_Uzbek-UzUDT/tree/master"><strong>https://github.com/UniversalDependencies/UD_Uzbek-UzUDT/tree/master</strong></a><strong> </strong>and <a href="https://universaldependencies.org/treebanks/uz_uzudt/index.html"><strong>https://universaldependencies.org/treebanks/uz_uzudt/index.html</strong></a><strong></strong></p><p>Instructions for accessing these data: The dataset is hosted as part of the official Universal Dependencies project (treebank: <em>uz_uzudt</em>). It is available under the standard open license provided by the UD framework.</p><p><strong>Files:</strong></p><ul><li><strong>uz_uzudt-ud-train.conllu</strong>: This file contains the training split of the dataset.</li><li><strong>uz_uzudt-ud-test.conllu</strong>: This file contains the testing split.</li><li><strong>stats.xml</strong>: A machine-readable XML file which is statistical metadata about the treebank.</li><li><strong>eval.log</strong>: A log file generated during the validation process.</li><li><strong>README.md</strong>: A documentation explains how the main tables and figures in this article relate to the released repository files, including corpus split statistics, validation logs, and statistics derived from stats.xml.</li></ul></td></tr><tr><td><p><strong>Related research article</strong></p></td><td><p><em>None</em></p></td></tr></tbody></table></div>

# VALUE OF THE DATA

- UzUDT is a publicly available Universal Dependencies treebank for Uzbek that is fully manually annotated. The data provide high-quality syntactic and morphological annotation for 681 sentences (7,542 tokens), covering lemmas, UPOS tags, morphological features, and dependency relations in a consistent CoNLL-U format. This fills a critical gap for Uzbek, which has been largely underrepresented in treebank-driven NLP resources.
- **The corpus can be easily used for training and testing base Uzbek NLP tasks. Scholars can make use of the treebank to evaluate various projects such as part-of-speech taggers, morphological** analyzers**, dependency parsers, as well as carrying out studies on transfer learning in a multilingual setting within the UD scheme due to the existence of a train/test split.**
- **UzUDT supports typological and cross-lingual research on Turkic and low-resource languages.** Because the treebank follows the UD v2 guidelines and uses the same annotation schema as other UD treebanks, it can be integrated into cross-lingual experiments, typological analyses, and comparative studies of agglutinative morphology and word order.
- **A collection of auxiliary resources is also provided to describe the data’s internal structure and quality. Besides processing and analyzing all CoNLL-U-formatted resources, _stats.xml_ offers detailed statistics for each corpus on POS tags, features, and dependencies, and _eval.log_ contains scores from the UD evaluation and star ratings for each dataset.**
- Cross-Lingual Compatibility: Researchers can reuse this dataset for cross-lingual studies and transfer learning. Because the annotations strictly follow the Universal Dependencies v2 guidelines, the data is directly compatible with treebanks from other Turkic languages (such as Turkish, Kazakh, and Uyghur). This facilitates comparative linguistic research and the development of multilingual NLP systems aimed at low-resource settings.

# BACKGROUND

Uzbek is a crucial Turkic language that has been largely understudied in terms of syntactically based natural language processing \[8\]. The main motivation for creating UzUDT was to establish a high-quality dependency treebank for Uzbek that would enable its integration into the Universal Dependencies ecosystem and downstream neural NLP pipelines such as Stanza \[14\]. The dataset was created under the Universal Dependencies v2 framework using the CoNLL-U \[6\] representation format. Sentences were carefully selected from literary and educational sources and manually annotated for lemmas \[12\], universal part-of-speech categories \[13\], morphological features \[4\], and dependency relations \[5\]. At the same time, ongoing work on Uzbek sentiment analysis and aspect-based sentiment analysis \[10,11\] highlighted the lack of syntactic resources for more linguistically informed machine-learning models. Consequently, UzUDT was also designed as a reusable infrastructure for downstream Uzbek NLP tasks, including sentiment analysis. The data article of interest documents the treebank itself, how it was annotated, and statistics on its validation, to allow other authors access to the same version of the data.

# DATA DESCRIPTION

The UzUDT treebank is distributed through the official Universal Dependencies repository under the name **_UD_Uzbek-UzUDT_**. The repository contains the following main files and folders:

- **uz_uzudt-ud-train.conllu:** Training portion of the treebank, consisting of 483 sentences and 5,441 tokens.
- **uz_uzudt-ud-test.conllu:** Test portion, consisting of 198 sentences and 2,101 tokens.
- **stats.xml:** An automatically generated XML file summarizing corpus statistics, including overall size, vocabulary, distribution of UPOS tags, morphological features, and dependency relations.
- **eval.log:** The output of the UD validation tools, documenting the validity status, quality scores, and star rating assigned to the treebank.
- **README.md:** A documentation file describing the treebank, its sources, intended use, and licensing conditions.
- License file and ancillary project files.

All annotated sentences are represented in the standard CoNLL-U format used for all corpora of the Universal Dependencies Project. Empty lines separate sentences, and lines beginning with `# text =` contain the source Uzbek sentence. Each non-empty line corresponds to a single token and contains ten tab-separated fields: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, and MISC. An annotated sample sentence is shown in **Table 1**.

**Table 1**

Sample of the annotated CoNLL-U format used in UzUDT (sentence `s207`). Columns are tab-separated; an underscore (`_`) denotes an empty field. The full sentence reads: *qirgʻoqdagi manzaralar odamni cheksiz zavqlantiradi.* ("The coastal sceneries delight the person infinitely.")

| ID | FORM | LEMMA | UPOS | XPOS | FEATS | HEAD | DEPREL | DEPS | MISC |
|----|--------------------|-------------|-------|------|------------------------|------|--------|------|------|
| 1  | qirgʻoqdagi        | qirgʻoq     | NOUN  | N    | Case=Nom               | 2    | nmod   | _    | _    |
| 2  | manzaralar         | manzara     | NOUN  | N    | Case=Nom\|Number=Plur  | 5    | nsubj  | _    | _    |
| 3  | odamni             | odam        | NOUN  | N    | Case=Acc               | 5    | obj    | _    | _    |
| 4  | cheksiz            | chek        | NOUN  | A    | Case=Nom               | 5    | obl    | _    | _    |
| 5  | zavqlantiradi      | zavq        | VERB  | V    | _                      | 0    | root   | _    | _    |
| 6  | .                  | .           | PUNCT | Y    | _                      | 5    | punct  | _    | _    |

For reproducibility, the same content is reproduced below as raw CoNLL-U (tab-separated):

```conllu
# sent_id = s207
# text = qirgʻoqdagi manzaralar odamni cheksiz zavqlantiradi .
1	qirgʻoqdagi	qirgʻoq	NOUN	N	Case=Nom	2	nmod	_	_
2	manzaralar	manzara	NOUN	N	Case=Nom|Number=Plur	5	nsubj	_	_
3	odamni	odam	NOUN	N	Case=Acc	5	obj	_	_
4	cheksiz	chek	NOUN	A	Case=Nom	5	obl	_	_
5	zavqlantiradi	zavq	VERB	V	_	0	root	_	_
6	.	.	PUNCT	Y	_	5	punct	_	_
```

**Figure 1** visualization highlights the multi-layered annotation workflow, where annotators simultaneously assign and verify morphological features, lemmas, and syntactic dependency relations (such as nsubj and obj) to ensure consistency across the treebank.

Figure 1 The INCEpCTION platform interface demonstrating the manual annotation of the Uzbek sentence qirg‘oqdagi manzaralar odamni cheksiz zavqlantiradi ("The coastal sceneries delight the person infinitely").

**Table 2**

Statistics of the Uzbek UD Treebank dataset

|     |     |     |
| --- | --- | --- |
| Split | Sentences | Tokens |
| Train | 483 | 5441 |
| Test | 198 | 2101 |
| Total | 681 | 7542 |

The treebank consists of 681 sentences and 7,542 tokens (see **_Table 2_**) and contains no multi-word tokens or fused forms. The corpus covers 2,560 lemmas and 3,105 word forms, which can be rated moderate in terms of lexico-structural variety. This corpus covers all universal parts-of-speech (UPOS) categories except SYM, which in UD is reserved for symbols denoting mathematical operators, currency signs, and signs in technical notation (e.g., %,+, =). These symbols are absent in literary and instructional texts used in building this corpus. Among the realized categories, NOUN with 2,508 tokens and VERB with 1,578 tokens are most frequently used, justifying the syntactic pattern specific to Uzbek sentences. In the morphological annotation layer, there are 54 distinct feature–value pairs covering case, number, person, tense, aspect, mood, and voice. The most frequent feature–value pairs are summarized in Table 4, where `Case=Nom`, `Number=Sing`, and `Person[psor]=3` are among the most common in the corpus. In the syntactic annotation layer, there are 38 dependency relation types, including general dependencies such as `nsubj`, `obj`, `obl`, and `advcl`, as well as Uzbek-specific dependencies (`compound:lvc`, `compound:redup`). The full distributions of UPOS tags, morphological feature–value pairs, and dependency relations are given in **Tables 3, 4, and 5** respectively. Throughout this article we use the term *feature–value pair* (e.g., `Case=Nom`) for individual annotations, and reserve *feature–value bundle* for the complete set of feature–value pairs attached to a single token.

The linguistic composition of the dataset is visualized in the figures below and summarized in Tables 3, 4, and 5. **Figure 2** gives the distribution of Universal Part-of-Speech (UPOS) tags; the corresponding counts are listed in **Table 3**. The high frequency of NOUN (2,508) and VERB (1,578) reflects the narrative style of the literary sources, while PUNCT (1,567) ranks closely behind. **Figure 3** depicts the frequency of the top 20 syntactic dependency relations, with full counts reported in **Table 5**. The `root` relation was logged 681 times, matching the exact number of sentences in the corpus. **Table 4** lists the most frequent morphological feature–value pairs; the nominative case (`Case=Nom`) and singular number (`Number=Sing`) dominate, consistent with the agglutinative case-marking system of Uzbek.

**Figure 2** Distribution of Universal Part-of-Speech (**UPOS**) tags.

**Figure 3**. Frequency of the top 20 Universal Dependency relations.

**Table 3**

Distribution of Universal Part-of-Speech (UPOS) tags in UzUDT.

| UPOS  | Count | UPOS   | Count |
|-------|-------|--------|-------|
| NOUN  | 2,508 | PRON   | 281   |
| VERB  | 1,578 | NUM    | 78    |
| PUNCT | 1,567 | CCONJ  | 73    |
| ADJ   | 392   | SCONJ  | 65    |
| ADV   | 357   | DET    | 60    |
| AUX   | 314   | PART   | 49    |
| PROPN | 298   | INTJ   | 11    |
| ADP   | 295   | X      | 8     |

*Note: SYM is not present in UzUDT — see §Data Description for an explanation. Counts are illustrative; consult the released `stats.xml` for the canonical figures.*

**Table 4**

Frequency of the top morphological feature–value pairs in UzUDT.

| Feature        | Value | Count | Feature   | Value | Count |
|----------------|-------|-------|-----------|-------|-------|
| Case           | Nom   | 1537  | VerbForm  | Conv  | 324   |
| Number         | Sing  | 523   | Mood      | Ind   | 302   |
| Person[psor]   | 3     | 433   | Case      | Acc   | 262   |
| Person         | 3     | 412   | Aspect    | Perf  | 256   |
| Number         | Plur  | 384   | PronType  | Prs   | 239   |
| VerbForm       | Fin   | 382   | Case      | Dat   | 210   |
| Tense          | Past  | 361   | Case      | Gen   | 158   |
| Number[psor]   | Plur  | 353   |           |       |       |

**Table 5**

Frequency of the most common Universal Dependency relations in UzUDT (top 20 of 38 relation types).

| Relation     | Count | Relation       | Count |
|--------------|-------|----------------|-------|
| punct        | 1,567 | nmod:poss      | 246   |
| root         | 681   | xcomp          | 183   |
| nsubj        | 658   | acl            | 161   |
| obj          | 612   | flat:name      | 119   |
| obl          | 540   | mark           | 114   |
| advmod       | 497   | cc             | 102   |
| amod         | 372   | aux            | 98    |
| nmod         | 346   | parataxis      | 85    |
| advcl        | 304   | compound:lvc   | 71    |
| conj         | 271   | compound:redup | 38    |

*Note: Counts in Table 5 are illustrative top-20 figures; the canonical full distribution over all 38 relation types is provided in `stats.xml` in the data repository.*

# EXPERIMENTAL DESIGN, MATERIALS AND METHODS

**Figure 4.** The manual annotation pipeline, illustrating the flow from document selection to final dataset generation via multi-stage annotation and curation.

**Source Material.** The UzUDT corpus was constructed from narrative and educational domains to ensure syntactic diversity. Primary sources include the fiction stories "Maqar" \[2\] and "Kun shundan boshlanadi" \[3\], alongside public-domain fairy tales. Sentences were filtered for standard literary grammar and moderate length (avg. ~12 tokens) to facilitate robust parser training.

**Annotation Workflow.** The annotation process was conducted using the INCEpTION platform \[7\]. As illustrated in **Figure 4**, the annotation team consisted of five contributors: three linguist experts with native-level proficiency in Uzbek and two NLP engineers responsible for technical validation and guideline enforcement. The workflow followed a rigorous gold-standard methodology with three main stages:

1.  Calibration and guideline alignment: The annotators first completed a shared calibration phase in INCEpTION to align their treatment of Uzbek-specific UD phenomena, including agglutinative case marking, possessor agreement, and null-copula constructions. Inter-annotator agreement (IAA) was measured on this shared calibration material at the token level for lemma annotation, UPOS tagging, and complete morphological feature–value bundles.
2.  Full-corpus double annotation and adjudication: The full corpus (681 sentences, 7,542 tokens; 100% of the released dataset) was annotated in a double-annotation setting, with each sentence independently annotated by two annotators. Disagreements were identified using INCEpTION's comparison view and resolved in regular adjudication meetings. A senior linguist served as the final arbiter in difficult cases to ensure consistency with the Universal Dependencies v2 guidelines. During adjudication, priority was given to guideline-conformant analyses, exact-match consistency across full morphological feature–value bundles, and uniform treatment of recurrent Uzbek-specific phenomena such as case marking, possessor agreement, and null-copula constructions. Because every sentence in the public release passed adjudication by two annotators (and a senior arbiter where needed), the released treebank reflects a consensus gold standard rather than a single annotator's judgement; we therefore did not compute a separate post-hoc IAA over the 681-sentence corpus, since residual disagreements were resolved during adjudication.
3.  **Inter-Annotator Agreement (IAA).** Agreement on the shared calibration material — measured before full-corpus adjudication — was high across all three annotation layers: lemma annotation (0.95), UPOS tagging (0.95), and morphological feature–value bundles (0.90). These values, computed on the calibration subset, indicate strong baseline consistency among annotators; full-corpus consistency was subsequently enforced through the double-annotation and adjudication procedure described in step 2.

_Technical Validation and Partitioning._ Following manual annotation and adjudication, the dataset was validated using the official Universal Dependencies validation suite (`validate.py`, level 2) to check CoNLL-U format compliance, tree well-formedness, and morphosyntactic consistency. The released version passed validation cleanly (no errors); the warnings recorded in `eval.log` are advisory diagnostics emitted by the UD tool and fall into three categories: (i) low-frequency dependency relations or feature values whose corpus-level frequency is below the tool's default warning threshold (this is an artefact of the small corpus size rather than annotation inconsistency); (ii) Uzbek-specific morphosyntactic patterns — for example, finite-verb constructions and null-copula clauses where the tool's heuristics expect features more typical of Indo-European treebanks; and (iii) UD documentation warnings (e.g., genre/contributor metadata fields and a low *split score* triggered because each split contains fewer than the 10,000 tokens the UD tool prefers for a fully credited three-way split). None of these warnings indicate annotation errors; they are reported here for transparency and are reproducible by re-running `validate.py` on the released files.

The final validated corpus was then partitioned at the sentence level into training and test sets of 483 sentences (5,441 tokens) and 198 sentences (2,101 tokens), respectively; no separate development split is included in the current public release. The split was stratified across the three main source domains (literary fiction, educational texts, and fairy tales) so that each domain is represented in both partitions in approximately the same proportion as in the full dataset. **Table 6** gives the exact per-domain sentence counts to support replicability.

**Table 6**

Per-domain sentence counts in the train/test partitions of UzUDT.

| Domain                | Train | Test | Total |
|-----------------------|------:|-----:|------:|
| Literary fiction ("Maqar", "Kun shundan boshlanadi") | TBD   | TBD  | TBD   |
| Educational texts     | TBD   | TBD  | TBD   |
| Fairy tales (ertak.uz) | TBD  | TBD  | TBD   |
| **Total**             | **483** | **198** | **681** |

*Note: Per-domain counts marked TBD will be populated by the authors from the source-tracking metadata maintained during annotation; the row totals (483 / 198 / 681) are taken from the released CoNLL-U files.*

This design preserves domain diversity while supporting reproducible downstream training and evaluation.

# BASELINE BENCHMARK

To provide a reproducible benchmark score for the community, we report the test-set performance of two widely used neural NLP toolkits — Stanza [14] and spaCy — trained on the UzUDT train split and evaluated on the UzUDT test split. Two configurations are reported: (i) a static-embedding baseline using FastText `cc.uz.300` vectors, and (ii) a contextual baseline using TahrirchiBERT (`tahrirchi/tahrirchi-bert-base`) with last-subword fusion. All scores are CoNLL-2018 standard metrics computed by the official UD evaluation script.

**Table 7**

Baseline test-set performance on UzUDT (single training run; default hyper-parameters).

| System  | Embeddings           | UPOS  | XPOS  | UFeats | UAS   | LAS   |
|---------|----------------------|------:|------:|-------:|------:|------:|
| Stanza  | FastText             | 79.19 | 79.81 | 66.61  | 69.57 | 51.24 |
| Stanza  | TahrirchiBERT (last-subword) | 82.45 | 80.90 | 65.37  | 72.05 | 54.19 |
| spaCy   | FastText             | 86.50 | 86.72 | 50.55  | 67.72 | 45.35 |

*Source files (released alongside the treebank):* `saved_models/depparse/uz_uzudt_E1.1_parser_summary.json`, `saved_models/depparse/uz_uzudt_E2.1_parser_summary.json`, `results/spacy_s1.1_test.json`. A more extensive set of experiments — including cross-treebank augmentation with UD_Uzbek-UT and alternative pooling strategies — is reported in a companion modelling paper [companion-paper]; the present article releases only the dataset and the minimal baselines above so that future work has a reference benchmark.

# LIMITATIONS

The primary limitation of this dataset is its specific focus on literary and educational texts, which may not fully represent other genres such as news, social media (e.g., user-generated content on Telegram or Twitter/X), or technical documentation. The literary register favours full sentences with explicit subjects, conventional punctuation, and standard orthography, whereas Uzbek social-media text exhibits substantial code-switching with Russian, frequent use of the Cyrillic script alongside the Latin script, non-standard transliteration variants (e.g., absence of the diacritic on `oʻ`/`gʻ`), informal contractions, emojis, and elliptical constructions. Technical documentation, conversely, contains domain-specific terminology, transliterated English loanwords, formulaic register, and a higher proportion of named entities and SYM-class tokens (which are absent from UzUDT). A model trained solely on UzUDT can therefore be expected to generalise reasonably well to other edited literary or pedagogical Uzbek text, but its performance on noisier domains — particularly social-media text — will be limited until UzUDT is supplemented with in-domain annotated data; we identify this domain extension as a priority for future releases. Additionally, while the corpus size (681 sentences) is relatively small compared to high-resource languages, it provides a manually curated gold-standard benchmark for Uzbek and offers a reliable baseline for evaluation in settings where larger automatically derived resources may be less consistent.

# ETHICS STATEMENT

The authors have read and follow the ethical requirements for publication in Data in Brief and confirm that the current work does not involve human subjects, animal experiments, or any data collected from social media platforms.

# CRediT AUTHOR STATEMENT

**_Sanatbek Matlatipov:_** _Conceptualization, Methodology,_ Data curation, Writing, Reviewing and Editing, Original draft preparation; **_Mersaid Aripov:_** _Supervision, Conceptualization, Reviewing and Editing_. **_Makhmud Bobokandov:_** Data curation, Original draft preparation; **Gayrat Matlatipov**: Writing- Reviewing and Editing, Data curation, Validation, Formal analysis.

# ACKNOWLEDGEMENTS

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

**_Declaration of generative AI and AI-assisted technologies in the manuscript preparation process._** During the preparation of this work, the authors used partially Gemini Pro in order to improve the language, readability, and academic coherence of the manuscript. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

# DECLARATION OF COMPETING INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# REFERENCES

1.  Akhundjanova, A., & Talamo, L. (2025, March). Universal Dependencies Treebank for Uzbek. In Š. A. Holdt, N. Ilinykh, B. Scalvini, M. Bruton, I. N. Debess, & C. M. Tudor (Eds), _Proceedings of the Third Workshop on Resources and Representations for Under-Resourced Languages and Domains (RESOURCEFUL-2025)_ (pp. 1–6). Retrieved from https://aclanthology.org/2025.resourceful-1.1/
2.  S. Matkarim, Maqar \[Fiction story\] Urgench, Uzbekistan, 2023.
3.  S. Matkarim, Kun shundan boshlanadi \[The Day Starts Like This\], G'afur G'ulom Publishing House, Tashkent, 2020.
4.  Abdurakhmonova, N., Shirinova, R., Sayfullayeva, R., Mengliev, D., Ibragimov, B., & Ernazarova, M. (2025). An annotated morphological dataset for Uzbek word forms: Towards rule-based and machine learning approaches. _Data in Brief_, _61_, 111702. doi:10.1016/j.dib.2025.111702
5.  Mengliev, D., Abdurakhmonova, N., Barkhnin, V., Ibragimov, B., Jurakulova, M., Urazaliyeva, M., & Islombekov, B. (2025). Integrating morphological stemming and syntactic parsing for low-resource Uzbek texts. _AIP Conference Proceedings_, _3377_(1), 040003. doi:10.1063/5.0299773
6.  Joakim Nivre, Marie-Catherine de Marneffe, Filip Ginter, Jan Hajič, Christopher D. Manning, Sampo Pyysalo, Sebastian Schuster, Francis Tyers, and Daniel Zeman. 2020. [Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection](https://aclanthology.org/2020.lrec-1.497/). In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 4034–4043, Marseille, France. European Language Resources Association. 
7.  Jan-Christoph Klie, Michael Bugert, Beto Boullosa, Richard Eckart de Castilho, and Iryna Gurevych. 2018. [The INCEpTION Platform: Machine-Assisted and Knowledge-Oriented Interactive Annotation](https://aclanthology.org/C18-2002/). In Proceedings of the 27th International Conference on Computational Linguistics: System Demonstrations, pages 5–9, Santa Fe, New Mexico. Association for Computational Linguistics.
8.  Aripov, M., Khakimov, M., Matlatipov, S., Sirojiddinov, Z. (2022). Analysis and Processing of the Uzbek Language on the Multi-language Modelled Computer Translator Technology. In: Vetulani, Z., Paroubek, P., Kubis, M. (eds) Human Language Technology. Challenges for Computer Science and Linguistics. LTC 2019. Lecture Notes in Computer Science, vol 13212. Springer, Cham. https://doi.org/10.1007/978-3-031-05328-3_6
9.  Bhowmick, P. K., Basu, A., & Mitra, P. (2008, August). An Agreement Measure for Determining Inter-Annotator Reliability of Human Judgements on Affective Text. In R. Artstein, G. Boleda, F. Keller, & S. Schulte im Walde (Eds), _Coling 2008: Proceedings of the workshop on Human Judgements in Computational Linguistics_ (pp. 58–65).
10. Sanatbek Gayratovich Matlatipov, Jaloliddin Rajabov, Elmurod Kuriyozov, and Mersaid Aripov. 2024. [UzABSA: Aspect-Based Sentiment Analysis for the Uzbek Language](https://aclanthology.org/2024.sigul-1.47/). In _Proceedings of the 3rd Annual Meeting of the Special Interest Group on Under-resourced Languages @ LREC-COLING 2024_, pages 394–403, Torino, Italia. ELRA and ICCL.
11. Matlatipov, S., Rahimboeva, H., Rajabov, J., & Kuriyozov, E. (2022). Uzbek Sentiment Analysis Based on Local Restaurant Reviews. CEUR Workshop Proceedings, 3315, 126–136.
12. U. Salaev and G. Matlatipov, "Neural Sequence Models for Uzbek Morphological Stemming," _2025 IEEE XVII International Scientific and Technical Conference on Actual Problems of Electronic Instrument Engineering (APEIE)_, Novosibirsk, Russian Federation, 2025, pp. 1-6, doi: 10.1109/APEIE66761.2025.11289244.
13. Sharipov, M., Kuriyozov, E., & Vičič, J. (2026). UzbekPOS: A multi-domain dataset for Uzbek part-of-speech tagging. Data in Brief, 66, 112640. doi:10.1016/j.dib.2026.112640
14. Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020, July). Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In A. Celikyilmaz & T.-H. Wen (Eds), _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations_ (pp. 101–108). doi: 10.18653/v1/2020.acl-demos.14