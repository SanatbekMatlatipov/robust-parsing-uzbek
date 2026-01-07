from pathlib import Path
import stanza
from stanza.utils.conll import CoNLL

# Paths relative to stanza-train root
GOLD_TEST = "data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.conllu"
OUT_SYS   = "data/udbase/UD_Uzbek-UzUDT/uz_uzudt-ud-test.system.conllu"

DEPPARSE_MODEL   = "saved_models/depparse/uz_uzudt_nocharlm_parser.pt"
DEPPARSE_PRETRAIN = "wordvec/uz/pretrain/fasttext_cc_uz_300.pt"

# 1️⃣ Load the gold CoNLL-U file as a Stanza Document
doc_gold = CoNLL.conll2doc(GOLD_TEST)

# 2️⃣ Build a pipeline that only runs the dependency parser
#    on an already-tokenized & pretagged document.
nlp = stanza.Pipeline(
    lang="uz",
    processors="depparse",
    depparse_model_path=DEPPARSE_MODEL,
    depparse_pretrain_path=DEPPARSE_PRETRAIN,  # <-- tell it where the .pt is
    depparse_pretagged=True,                   # <-- use existing tokens/tags
    allow_unknown_language=True
)

# 3️⃣ Run the parser
doc_sys = nlp(doc_gold)

# 4️⃣ Write system predictions to CoNLL-U
with open(OUT_SYS, "w", encoding="utf-8") as f:
    CoNLL.write_doc2conll(doc_sys, f)

print("✅ Wrote system parse to:", OUT_SYS)
