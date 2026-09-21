# Speaker Notes — *Towards Robust Uzbek Neural Dependency Parsing*

**Event:** NLDB 2026 · 18 June 2026 · 15-minute talk
**Speaker:** Sanatbek Matlatipov (National University of Uzbekistan)
**Deck:** `presentation.pdf` (23 slides incl. 3 appendix backups)

**Delivery tips**
- Total spoken budget ≈ **13.5 min**, leaving ~1.5 min buffer for Q&A overflow.
- Numbers to land hard: **85.08 / 72.39 / 63.81** (UPOS/UAS/LAS) and **cross-treebank +9.62–11.16 LAS**.
- The two ideas the audience must remember: **last-subword fusion** and **cross-treebank training**.
- Slides 12 (fusion) and 16/17 (results) are the core — slow down there. Speed up on 6 and 15.
- Click cues: only Slide 12 has a build (the orange "pick last" arrow appears on the 2nd click — *if you kept the static version, no clicks needed*).

---

## Slide 1 — Title  ·  ~20 s
- "Good morning. I'm Sanatbek Matlatipov from the National University of Uzbekistan."
- "My talk is about making Uzbek dependency parsing *robust* in a genuinely low-resource setting — using two lightweight, efficient ideas rather than bigger models."
- *Transition:* "Here's where we're going."

## Slide 2 — Outline  ·  ~15 s
- "Quick motivation, the dataset we built, our method, then results and takeaways."
- Don't read every word — point and move on.

## Slide 3 — Why is Uzbek parsing hard?  ·  ~60 s
- "Three things make this hard. First, Uzbek is **agglutinative** — a single word stacks suffixes for case, tense, person, even evidentiality."
- "Second, **flexible word order** plus strong genre variation."
- "Third, it's **low-resource** — fewer than a thousand gold UD sentences exist."
- Point to the block: "Take *bolalarning* — 'of the children.' The stem *bola* plus plural *-lar* plus genitive *-ning*. That final suffix **is** the syntactic relation. Miss the suffix, miss the dependency."
- Gesture to the tree on the right: "This is a gold tree from our treebank."
- *Transition:* "So what exactly is missing?"

## Slide 4 — The gap and our approach  ·  ~45 s
- "The core problem: existing Uzbek treebanks are small, and neural parsers need **both** more data **and** better representations."
- "We attack both cheaply. On the data side, a new treebank — **UzUDT**, 684 gold sentences, the largest for Uzbek."
- "On the modeling side, two lightweight strategies: **last-subword fusion** and **cross-treebank training**."
- "Crucially — no heavy architecture changes. This is the efficient, low-resource-friendly path."
- *Transition:* "Concretely, four contributions."

## Slide 5 — Contributions  ·  ~50 s
- "One: the **UzUDT** treebank — built in INCEpTION with six annotators and full adjudication."
- "Two: **last-subword fusion**, a linguistically-motivated way to align BERT WordPieces to UD tokens."
- "Three: **cross-treebank training** — merging two genre-complementary Uzbek treebanks."
- "Four: a clean **factorial study** — three configs times two data settings, six runs — plus released code and twelve checkpoints."
- Land the box: "Best system: **85.08 UPOS, 72.39 UAS, 63.81 LAS** — and cross-treebank alone adds nine to eleven LAS points."
- *Transition:* "Briefly, where this sits in the literature."

## Slide 6 — Related work  ·  ~40 s  *(go fast)*
- "We build on Universal Dependencies and on morphology-aware parsing for Turkic languages like Turkish."
- "For Uzbek specifically, there was one small treebank, UD\_Uzbek-UT; we add UzUDT."
- "Architecturally we use a standard neural pipeline — Stanza-style, with a Dozat–Manning biaffine parser — and Uzbek encoders like TahrirchiBERT."
- "Our niche: **lightweight** robustness — suffix-aware fusion plus cross-treebank data, not a new architecture."
- *Transition:* "Let me start with the resource we built."

## Slide 7 — UzUDT treebank  ·  ~55 s
- "UzUDT: 684 sentences, about 7,800 tokens, across fiction and academic text."
- "Annotated in INCEpTION by six people — four linguists and two NLP engineers — with double annotation and **full adjudication**."
- "Full UD v2 layers: UPOS, XPOS, lemma, features, and dependency relations."
- Point to the table: "Inter-annotator agreement is strong — kappa of 95 on lemmas, 94 on UPOS, 91 on features. So the labels are reliable."
- "To our knowledge this is the **largest publicly available Uzbek UD treebank**."
- *Transition:* "We don't use it alone — we pair it with a second treebank."

## Slide 8 — Second treebank & data settings  ·  ~45 s
- "The second treebank is UD\_Uzbek-UT — 500 sentences, news and fiction, semi-automatic with manual correction."
- "That gives two data settings. Setting **.1** is UzUDT only — 451 training sentences. Setting **.2** merges both — 781."
- "So merging nearly **doubles** the training data. That's a direct test: is raw data quantity the bottleneck?"
- *Transition:* "But quantity isn't the whole story — the genres are complementary."

## Slide 9 — Why the treebanks are complementary  ·  ~55 s
- "These two treebanks cover **different syntax**, which is exactly why merging helps."
- Walk the bars: "UT is news-heavy, so it has many proper nouns — 5.2% versus our 0.3%."
- "UzUDT, being fiction and academic, has more pronouns and **four times** more adverbial clauses."
- "And UT has **ten times** more light-verb compounds."
- "So they fill each other's syntactic gaps — merging isn't just more data, it's **broader** data."
- *Transition:* "Now the model that consumes this data."

## Slide 10 — Pipeline overview  ·  ~55 s
- "Here's the full pipeline, left to right. A UD-tokenized sentence goes into a **frozen encoder** — FastText or TahrirchiBERT."
- "Then our **fusion** step maps subwords to UD tokens. Then a **joint tagger** predicts UPOS, XPOS, features. Then a **biaffine parser** produces the labeled tree."
- "Two design choices matter. The encoder is **frozen** — efficient. And training is **sequential**: tagger first, then the parser runs on **re-tagged** data — the dashed arrow — so there's no gold-tag leakage at inference."
- *Transition:* "Let me unpack each box, starting with how a token is represented."

## Slide 11 — Token representation: four channels  ·  ~45 s
- "Each token is a concatenation of four vectors: a static **word** embedding, a **character**-LSTM vector, the **predicted UPOS** embedding, and a **contextual** vector from BERT."
- "Two of the six runs differ here. In the **FastText** config, the word channel is static 300-d and there's **no** contextual channel."
- "In the **TahrirchiBERT** config, that contextual channel is a 768-d **fused** vector — and that fusion is our key contribution."
- *Transition:* "So how do we build that fused vector?"

## Slide 12 — Subword→token fusion (KEY)  ·  ~75 s  *(slow down)*
- "This is the heart of the method. BERT emits **WordPieces**, but UD annotates **tokens** — they don't line up. We need to collapse several subword vectors into one per token."
- Point to *bolalarning* splitting into *bola*, *##lar*, *##ning*.
- "Our heuristic — **last-subword**: take the vector of the **final** WordPiece as the token's representation. Here that's *##ning*, the orange one."
- "Why the last one? Because Uzbek is **suffixing** — the rightmost piece carries the **outermost** morphology. *##ning* literally is the genitive case. That's the bit the parser most needs."
- Contrast right side: "The baseline is **mean pooling** — average all pieces. It's language-agnostic, but it **dilutes** exactly that suffix signal."
- "So this is a one-line change motivated purely by Uzbek morphology — and later you'll see it's worth almost four LAS points."
- *Transition:* "That representation feeds the tagger."

## Slide 13 — Joint morphosyntactic tagger  ·  ~50 s
- "The tagger is a shared **BiLSTM** over the token vectors, feeding **three softmax heads** — UPOS, XPOS, and morphological features — jointly."
- "Joint tagging lets the three layers share evidence — useful when data is scarce."
- "And only the **predicted** UPOS embedding — not the gold tag — is exported downstream to the parser. Again, no leakage."
- *Transition:* "Finally, the parser itself."

## Slide 14 — Deep biaffine parser  ·  ~55 s
- "Standard **Dozat–Manning** biaffine parser. The BiLSTM states — now including the predicted UPOS embedding — are projected by MLPs into **head** and **dependent** subspaces."
- "A biaffine attention then scores **every** head–dependent pair, and we decode a labeled tree over the UD tokens."
- Point to the example tree: "Like this — *asarlar bizga eshigini ochib beradi* — subject, oblique, the xcomp, the object, and the root."
- *Transition:* "Now — does any of this actually help? Here's our experimental design."

## Slide 15 — Experiment matrix  ·  ~45 s  *(go fast)*
- "Six runs, fully factorial. Rows vary three things."
- "**E1 vs E2** isolates the embedding — static FastText versus contextual BERT."
- "**E2 vs E3** isolates fusion — our last-subword versus mean pooling."
- "And **.1 vs .2** isolates data — single versus cross-treebank."
- "The green row, **E2.2**, is the full system: BERT, last-subword, merged data."
- *Transition:* "Here's what happens."

## Slide 16 — Main results  ·  ~60 s
- "Tagging on the left, parsing on the right; the green row is our full system, E2.2."
- "E2.2 wins **four of five** metrics — best UPOS at **85.08**, best features, best UAS **72.39**, best LAS **63.81**."
- "The only exception is XPOS, where mean pooling edges ahead — a minor, fine-grained-tag effect."
- "Notice the LAS column climbs steadily as we add contextual embeddings, then fusion, then data."
- *Transition:* "Three findings fall out of this."

## Slide 17 — Three findings  ·  ~60 s
- "One: **contextual beats static** — BERT over FastText gives nearly five UPOS points on merged data."
- "Two: **fusion matters for parsing**. Last-subword over mean pooling is **+3.76 LAS**. And look at the ordering: mean-pooled BERT actually *loses* to FastText; last-subword BERT wins. So a bad fusion can waste your encoder."
- "Three — the biggest lever — **cross-treebank training**: nine to eleven LAS points just from merging. For low-resource parsing, **broader data beats clever modeling**."
- Gesture to the chart: "You can see E2.2 standing tallest."
- *Transition:* "Where does the system still fail?"

## Slide 18 — Error analysis  ·  ~50 s
- "We hand-checked 100 test sentences from E2.2."
- "The errors concentrate in attachment: **prepositional/oblique attachment** at 28%, **coordination scope** at 22%, then subordinate clauses and compounds."
- "The dominant confusion is *obl* versus *nmod*, and coordination scope."
- Point to the tree: "Here's a missed **light-verb construction** — gold *compound:lvc* in blue, the model's wrong arcs dashed in red. These rare constructions are exactly what the smaller treebank under-covers."
- *Transition:* "To wrap up."

## Slide 19 — Conclusion  ·  ~45 s
- "Three takeaways. We released **UzUDT**, the largest gold Uzbek UD treebank."
- "**Cross-treebank training** is the dominant factor — up to eleven LAS points."
- "And **last-subword fusion** is a cheap, linguistically-motivated win for parsing — almost four LAS points."
- "Best system: **85.08 / 72.39 / 63.81**."
- "Limitations: small dev sets and some genre mismatch. Next: BERTbek and BERT-plus-FastText fusion."
- "Everything — data and twelve checkpoints — is on Hugging Face, code on GitHub."

## Slide 20 — Thank you  ·  ~15 s
- "Thank you — I'd be happy to take questions."
- Leave contact slide up.

---

## Appendix (backup — only if asked)
- **A1 Parser dynamics:** "Dev LAS/UAS over training — merged converges higher and more stably than UzUDT-only."
- **A2 Tagger dynamics:** "UPOS/XPOS/UFeats F1 across the six runs — E2.2 leads on most curves."
- **A3 Correct parse:** "A clean example the full system gets right end-to-end."

### Likely questions — quick answers
- *Why freeze the encoder?* Efficiency and stability on <1k sentences; fine-tuning overfits.
- *Why not mBERT/UDify?* We compare lightweight Uzbek-specific encoders; multilingual baselines are future work.
- *Is last-subword just for Uzbek?* It targets suffixing/agglutinative morphology — should transfer to other Turkic languages.
- *Dev sets are tiny (45/78) — noise?* Yes, a stated limitation; the +9–11 LAS data effect is far above that noise.
- *XPOS regression?* Mean pooling smooths fine-grained tags slightly; net effect across the other four metrics still favors last-subword.
