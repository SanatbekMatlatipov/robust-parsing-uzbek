# scripts/pos_predict_from_udbase.py
from pathlib import Path
import stanza
from stanza.utils.conll import CoNLL

GOLD_TEST = "data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.conllu"
OUT_SYS   = "data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.pos.system.conllu"

TOKENIZER_MODEL = "saved_models/tokenize/uz_uzudt_tokenizer.pt"
POS_MODEL       = "saved_models/pos/uz_uzudt-base_tagger.pt"

# Load gold as pretokenized doc
doc_gold = CoNLL.conll2doc(GOLD_TEST)

nlp = stanza.Pipeline(
    lang="uz",
    processors="tokenize,pos",
    tokenize_pretokenized=True,
    tokenize_model_path=TOKENIZER_MODEL,
    pos_model_path=POS_MODEL,
    allow_unknown_language=True
)

doc_sys = nlp(doc_gold)

# ✅ Write CoNLL-U using the helper (works across Stanza versions)
with open(OUT_SYS, "w", encoding="utf-8") as f:
    CoNLL.write_doc2conll(doc_sys, f)

print("✅ Wrote:", OUT_SYS)
