# scripts/conllu2txt.py
from conllu import parse_incr
from pathlib import Path

def conv(src, dst):
    out = []
    with open(src, encoding="utf-8") as f:
        for sent in parse_incr(f):
            toks = [t["form"] for t in sent if isinstance(t["id"], int)]
            out.append(" ".join(toks))
    Path(dst).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("Wrote", dst, f"({len(out)} sentences)")

base = "data/udbase/UD_Uzbek-UzUDT"
conv(f"{base}/uz_uzudt-ud-train.conllu", f"{base}/uz_uzudt-ud-train.txt")
conv(f"{base}/uz_uzudt-ud-dev.conllu",   f"{base}/uz_uzudt-ud-dev.txt")
conv(f"{base}/uz_uzudt-ud-test.conllu",  f"{base}/uz_uzudt-ud-test.txt")
