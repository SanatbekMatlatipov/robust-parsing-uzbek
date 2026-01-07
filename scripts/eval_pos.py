#!/usr/bin/env python3
"""
Evaluate POS tagging accuracy on CoNLL-U files.

Usage:
  python3 scripts/eval_pos.py <gold.conllu> <system.conllu> [--field upos|xpos]

Notes:
- Assumes sentence & token alignment between gold and system files.
- Skips multi-word-token lines (id like "3-4").
"""
import sys
import argparse
from conllu import parse_incr

def read_sents(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(parse_incr(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("system")
    ap.add_argument("--field", default="upos", choices=["upos", "xpos"],
                    help="Which POS field to evaluate (default: upos)")
    args = ap.parse_args()

    G = read_sents(args.gold)
    S = read_sents(args.system)

    if len(G) != len(S):
        raise ValueError(f"Sentence count mismatch: gold={len(G)} system={len(S)}")

    total = correct = 0
    for gs, ss in zip(G, S):
        g_tokens = [t for t in gs if isinstance(t["id"], int)]
        s_tokens = [t for t in ss if isinstance(t["id"], int)]
        if len(g_tokens) != len(s_tokens):
            raise ValueError("Token count mismatch in a sentence. "
                             "Ensure you evaluated on gold tokenization.")
        for gt, st in zip(g_tokens, s_tokens):
            gtag = gt.get(args.field)
            stag = st.get(args.field)
            if gtag is None:  # skip empty tags in gold
                continue
            total += 1
            correct += int(gtag == stag)

    acc = 100.0 * correct / total if total else 0.0
    print(f"{args.field.upper()} accuracy: {acc:.2f}%  ({correct}/{total})")

if __name__ == "__main__":
    main()
