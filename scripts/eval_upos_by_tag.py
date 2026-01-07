#!/usr/bin/env python3
"""
Per-tag POS accuracy on CoNLL-U files.

Usage:
  python3 scripts/eval_upos_by_tag.py <gold.conllu> <system.conllu> [--field upos|xpos]

Prints overall accuracy and per-tag accuracies (micro).
"""
import sys
import argparse
from collections import Counter
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
    by = Counter()
    cor_by = Counter()

    for gs, ss in zip(G, S):
        g_tokens = [t for t in gs if isinstance(t["id"], int)]
        s_tokens = [t for t in ss if isinstance(t["id"], int)]
        if len(g_tokens) != len(s_tokens):
            raise ValueError("Token count mismatch in a sentence."
                             " Ensure evaluation uses gold tokenization.")
        for gt, st in zip(g_tokens, s_tokens):
            gtag = gt.get(args.field)
            stag = st.get(args.field)
            if gtag is None:
                continue
            total += 1
            by[gtag] += 1
            is_correct = int(gtag == stag)
            cor_by[gtag] += is_correct
            correct += is_correct

    overall = 100.0 * correct / total if total else 0.0
    print(f"OVERALL {args.field.upper()}: {overall:.2f}%  ({correct}/{total})")
    for tag in sorted(by):
        acc = 100.0 * cor_by[tag] / by[tag]
        print(f"{tag:>6}: {acc:6.2f}%  ({cor_by[tag]}/{by[tag]})")

if __name__ == "__main__":
    main()
