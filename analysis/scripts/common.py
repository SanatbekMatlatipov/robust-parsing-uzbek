"""Shared helpers: paths, CoNLL-U input/output and text normalisation."""
import json
import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPTS)
RESULTS = os.path.join(BASE, "results")
REPO = os.path.abspath(os.path.join(BASE, ".."))

UZUDT = os.path.join(REPO, "data", "udbase", "UD_Uzbek-UzUDT")
UT = os.path.join(REPO, "data", "udbase", "UD_Uzbek-UT")
OLD_SPLIT = os.path.join(REPO, "data", "depparse")  # uz_uzudt.{train,dev,test}.in.conllu (.1 experiments)
MERGED = os.path.join(REPO, "data", "depparse", "merged")  # uz_combined.* (.2 experiments)

os.makedirs(RESULTS, exist_ok=True)


def read_conllu(path):
    """Sentences as dicts: {'comments': [...], 'rows': [[10 columns], ...]} (multiword/empty nodes kept as rows)."""
    sents, comments, rows = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if rows:
                    sents.append({"comments": comments, "rows": rows})
                comments, rows = [], []
            elif line.startswith("#"):
                comments.append(line)
            else:
                rows.append(line.split("\t"))
    if rows:
        sents.append({"comments": comments, "rows": rows})
    return sents


def write_conllu(sents, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in sents:
            for c in s["comments"]:
                f.write(c + "\n")
            for r in s["rows"]:
                f.write("\t".join(r) + "\n")
            f.write("\n")


def words(sent):
    """Syntactic words only (integer ids)."""
    return [r for r in sent["rows"] if r[0].isdigit()]


def sent_text(sent):
    return " ".join(r[1] for r in words(sent))


def sent_id(sent):
    for c in sent["comments"]:
        if c.startswith("# sent_id"):
            return c.split("=", 1)[1].strip()
    return None


APOSTROPHES = {"ʻ": "'", "‘": "'", "’": "'", "ʼ": "'", "`": "'"}


def normalise(text):
    """Comparison key: lower case, one apostrophe symbol."""
    for a, b in APOSTROPHES.items():
        text = text.replace(a, b)
    return text.lower()


def cp1252_mojibake(text):
    """Reproduce the corruption found in data/{pos,depparse}/merged/*: UTF-8 bytes decoded as Windows-1252,
    undefined bytes passed through as the same code point (0x81, 0x8d, 0x8f, 0x90, 0x9d)."""
    out = []
    for b in text.encode("utf-8"):
        try:
            out.append(bytes([b]).decode("cp1252"))
        except UnicodeDecodeError:
            out.append(chr(b))
    return "".join(out)


def save_json(obj, name):
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    return path
