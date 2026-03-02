#!/usr/bin/env python3
"""
Convert CoNLL-U files to spaCy DocBin format.

This script reads Universal Dependencies CoNLL-U files and produces
spaCy-compatible .spacy (DocBin) binary files for training.

It handles:
  - UPOS tags  (token.pos_)
  - XPOS tags  (token.tag_)
  - Morphological features (token.morph)
  - Dependency relations (token.dep_)
  - Dependency heads (token.head)
  - Lemmas (token.lemma_)
  - Multi-word tokens (MWT ranges are skipped; only sub-tokens are used)

Usage:
    python spacy_uzbek/convert_conllu.py \
        --input  data/pos/uz_uzudt.train.in.conllu \
        --output spacy_uzbek/data/uz_uzudt.train.spacy \
        --lang uz

    # Or convert all standard splits at once:
    python spacy_uzbek/convert_conllu.py --convert-all
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Register Uzbek language before importing spaCy blank
from spacy_uzbek.lang.uz import Uzbek  # noqa: F401

import spacy
from spacy.tokens import Doc, DocBin


def parse_conllu(filepath: str):
    """
    Parse a CoNLL-U file and yield one sentence at a time.

    Each sentence is a list of dicts with keys:
        id, form, lemma, upos, xpos, feats, head, deprel, deps, misc

    Multi-word token lines (e.g., "1-2") are skipped.
    Empty nodes (e.g., "1.1") are skipped.
    """
    sentence = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                if sentence:
                    yield sentence
                    sentence = []
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                continue
            tok_id = fields[0]
            # Skip multi-word token ranges (e.g., "1-2") and empty nodes (e.g., "1.1")
            if "-" in tok_id or "." in tok_id:
                continue
            sentence.append({
                "id": int(tok_id),
                "form": fields[1],
                "lemma": fields[2],
                "upos": fields[3],
                "xpos": fields[4],
                "feats": fields[5],
                "head": int(fields[6]) if fields[6] != "_" else 0,
                "deprel": fields[7],
                "deps": fields[8],
                "misc": fields[9],
            })
    if sentence:
        yield sentence


def conllu_to_docbin(input_path: str, output_path: str, lang: str = "uz"):
    """
    Convert a CoNLL-U file to a spaCy DocBin (.spacy) file.

    Parameters
    ----------
    input_path : str
        Path to the input .conllu file.
    output_path : str
        Path to the output .spacy file.
    lang : str
        Language code (default: "uz").
    """
    nlp = spacy.blank(lang)
    doc_bin = DocBin(store_user_data=True)

    n_sents = 0
    n_tokens = 0

    for sentence in parse_conllu(input_path):
        words = [tok["form"] for tok in sentence]
        spaces = []
        for i, tok in enumerate(sentence):
            # Default: space after each token unless SpaceAfter=No
            if "SpaceAfter=No" in tok.get("misc", ""):
                spaces.append(False)
            else:
                spaces.append(True)

        doc = Doc(nlp.vocab, words=words, spaces=spaces)

        # Set POS, tag, morph, lemma, dep, and head for each token
        heads = []
        deps = []
        for i, tok in enumerate(sentence):
            token = doc[i]
            token.pos_ = tok["upos"] if tok["upos"] != "_" else ""
            token.tag_ = tok["xpos"] if tok["xpos"] != "_" else ""
            token.lemma_ = tok["lemma"] if tok["lemma"] != "_" else token.text

            # Morphological features
            if tok["feats"] and tok["feats"] != "_":
                token.set_morph(tok["feats"])

            # Dependency head: CoNLL-U uses 1-based; 0 = root
            # spaCy uses token index (0-based); root points to itself
            head_idx = tok["head"]
            if head_idx == 0:
                # Root token: head is itself in spaCy
                heads.append(i)
            else:
                heads.append(head_idx - 1)  # Convert 1-based to 0-based

            dep = tok["deprel"] if tok["deprel"] != "_" else "dep"
            deps.append(dep)

        # Assign heads and deps in bulk via Doc constructor workaround
        for i in range(len(doc)):
            doc[i].head = doc[heads[i]]
            doc[i].dep_ = deps[i]

        # Note: sentence boundaries are inferred from the dependency parse
        # (root tokens mark sentence starts), so explicit is_sent_start
        # is not needed when heads/deps are set.

        doc_bin.add(doc)
        n_sents += 1
        n_tokens += len(words)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc_bin.to_disk(output_path)

    print(f"  Converted {input_path}")
    print(f"    -> {output_path}  ({n_sents} sentences, {n_tokens} tokens)")
    return n_sents, n_tokens


def convert_all():
    """Convert all standard CoNLL-U splits to spaCy format."""
    base = PROJECT_ROOT
    data_out = base / "spacy_uzbek" / "data"

    conversions = [
        # UzUDT single-treebank splits
        ("data/pos/uz_uzudt.train.in.conllu", "uz_uzudt.train.spacy"),
        ("data/pos/uz_uzudt.dev.in.conllu",   "uz_uzudt.dev.spacy"),
        ("data/pos/uz_uzudt.test.in.conllu",  "uz_uzudt.test.spacy"),
        # UT single-treebank splits
        ("data/pos/uz_ut-ud-train.in.conllu", "uz_ut.train.spacy"),
        ("data/pos/uz_ut-ud-dev.in.conllu",   "uz_ut.dev.spacy"),
        ("data/pos/uz_ut-ud-test.in.conllu",  "uz_ut.test.spacy"),
        # Merged UzUDT+UT splits
        ("data/pos/merged/uz_combined.train.in.conllu", "uz_combined.train.spacy"),
        ("data/pos/merged/uz_combined.dev.in.conllu",   "uz_combined.dev.spacy"),
        ("data/pos/merged/uz_combined.test.in.conllu",  "uz_combined.test.spacy"),
    ]

    print(f"Converting CoNLL-U files to spaCy DocBin format...")
    print(f"Output directory: {data_out}\n")

    total_sents = 0
    total_tokens = 0
    skipped = []

    for conllu_rel, spacy_name in conversions:
        conllu_path = base / conllu_rel
        spacy_path = data_out / spacy_name
        if not conllu_path.exists():
            skipped.append(str(conllu_rel))
            continue
        n_s, n_t = conllu_to_docbin(str(conllu_path), str(spacy_path))
        total_sents += n_s
        total_tokens += n_t

    print(f"\nDone! Total: {total_sents} sentences, {total_tokens} tokens")
    if skipped:
        print(f"Skipped (file not found): {', '.join(skipped)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CoNLL-U files to spaCy DocBin (.spacy) format"
    )
    parser.add_argument("--input", "-i", type=str,
                        help="Path to input CoNLL-U file")
    parser.add_argument("--output", "-o", type=str,
                        help="Path to output .spacy file")
    parser.add_argument("--lang", "-l", type=str, default="uz",
                        help="Language code (default: uz)")
    parser.add_argument("--convert-all", action="store_true",
                        help="Convert all standard splits (UzUDT, UT, merged)")
    args = parser.parse_args()

    if args.convert_all:
        convert_all()
    elif args.input and args.output:
        conllu_to_docbin(args.input, args.output, lang=args.lang)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
