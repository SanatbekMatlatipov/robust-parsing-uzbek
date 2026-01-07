from pathlib import Path
import stanza
from stanza.utils.conll import CoNLL

TEST_CONLLU = "data/uz_uzudt/uz_uzudt-ud-test.conllu"
OUT_CONLLU  = "data/uz_uzudt/uz_uzudt-ud-test.pos.system.conllu"

POS_MODEL = "/Users/sanatbek/code/uzudtevaluations/stanza-train/saved_models/pos/uz_uzudt-base_tagger.pt"

doc_gold = CoNLL.conll2doc(TEST_CONLLU)

nlp = stanza.Pipeline(
    lang="uz",
    processors="pos",
    tokenize_pretokenized=True,
    pos_model_path=POS_MODEL,
)

doc_sys = nlp(doc_gold)
Path(OUT_CONLLU).write_text(doc_sys.to_conll(), encoding="utf-8")
print("Wrote:", OUT_CONLLU)
