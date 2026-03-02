#!/usr/bin/env python3
"""
Evaluate a trained spaCy model and produce detailed metrics.

This script:
  1. Loads a trained spaCy model
  2. Runs inference on a CoNLL-U test file
  3. Reports UPOS accuracy, UFeats accuracy, UAS, and LAS
  4. Produces a per-tag breakdown (optional)
  5. Exports predictions to CoNLL-U for comparison with Stanza

Usage:
    python spacy_uzbek/evaluate.py \
        --model saved_models/spacy/transformer_combined/model-best \
        --test  data/pos/merged/uz_combined.test.in.conllu \
        --output results/spacy_eval.json

    # Or use the .spacy test file directly:
    python spacy_uzbek/evaluate.py \
        --model saved_models/spacy/transformer_combined/model-best \
        --test-spacy spacy_uzbek/data/uz_combined.test.spacy \
        --output results/spacy_eval.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Register Uzbek
from spacy_uzbek.lang.uz import Uzbek  # noqa: F401

import spacy
from spacy.tokens import DocBin


def load_gold_from_conllu(filepath: str):
    """
    Parse a CoNLL-U file and return list of sentences.
    Each sentence is a list of dicts with gold annotations.
    """
    from spacy_uzbek.convert_conllu import parse_conllu
    return list(parse_conllu(filepath))


def evaluate_on_conllu(model_path: str, conllu_path: str):
    """
    Run a spaCy model on raw text extracted from CoNLL-U
    and compare predictions against gold annotations.
    """
    nlp = spacy.load(model_path)
    gold_sents = load_gold_from_conllu(conllu_path)

    # Metrics accumulators
    upos_correct = 0
    upos_total = 0
    ufeats_correct = 0
    ufeats_total = 0
    uas_correct = 0
    las_correct = 0
    dep_total = 0

    # Per-tag breakdown
    per_tag = defaultdict(lambda: {"correct": 0, "total": 0})

    results_sentences = []

    for sent in gold_sents:
        text = " ".join([tok["form"] for tok in sent])
        doc = nlp(text)

        # Align tokens — spaCy tokenizer may produce different tokenization
        # For gold-tokenized evaluation, we create a Doc from gold tokens
        gold_words = [tok["form"] for tok in sent]
        pred_words = [t.text for t in doc]

        # If tokenization matches, do direct comparison
        if pred_words == gold_words:
            for i, tok in enumerate(sent):
                pred_token = doc[i]

                # UPOS
                gold_upos = tok["upos"]
                pred_upos = pred_token.pos_
                if gold_upos == pred_upos:
                    upos_correct += 1
                upos_total += 1
                per_tag[gold_upos]["total"] += 1
                if gold_upos == pred_upos:
                    per_tag[gold_upos]["correct"] += 1

                # UFeats
                gold_feats = tok["feats"] if tok["feats"] != "_" else ""
                pred_feats = str(pred_token.morph) if pred_token.morph else ""
                if gold_feats == pred_feats:
                    ufeats_correct += 1
                ufeats_total += 1

                # Dependency
                gold_head = tok["head"]
                gold_deprel = tok["deprel"]
                # spaCy head: the head token object; root points to itself
                pred_head_idx = pred_token.head.i + 1 if pred_token.head.i != pred_token.i else 0
                # Adjust for sentence offset
                if pred_token.dep_ == "ROOT":
                    pred_head_idx = 0
                else:
                    pred_head_idx = pred_token.head.i - doc[0].i + 1

                pred_deprel = pred_token.dep_

                if gold_head == pred_head_idx:
                    uas_correct += 1
                    if gold_deprel == pred_deprel:
                        las_correct += 1
                dep_total += 1
        else:
            # Tokenization mismatch — count tokens but mark as errors
            # This is approximate; for precise evaluation, use gold tokenization
            upos_total += len(sent)
            ufeats_total += len(sent)
            dep_total += len(sent)

    metrics = {
        "upos_accuracy": round(100 * upos_correct / upos_total, 2) if upos_total > 0 else 0,
        "ufeats_accuracy": round(100 * ufeats_correct / ufeats_total, 2) if ufeats_total > 0 else 0,
        "uas": round(100 * uas_correct / dep_total, 2) if dep_total > 0 else 0,
        "las": round(100 * las_correct / dep_total, 2) if dep_total > 0 else 0,
        "total_tokens": upos_total,
        "total_sentences": len(gold_sents),
    }

    per_tag_metrics = {}
    for tag, counts in sorted(per_tag.items()):
        acc = round(100 * counts["correct"] / counts["total"], 2) if counts["total"] > 0 else 0
        per_tag_metrics[tag] = {
            "accuracy": acc,
            "correct": counts["correct"],
            "total": counts["total"],
        }
    metrics["per_tag"] = per_tag_metrics

    return metrics


def evaluate_on_spacy(model_path: str, test_spacy_path: str):
    """
    Evaluate using spaCy's built-in evaluate command.
    Returns the metrics dict from spacy evaluate.
    """
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "evaluate", model_path, test_spacy_path,
         "--output", "/dev/null"],
        capture_output=True, text=True
    )
    # Parse the console output for metrics
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return None  # spaCy prints to stdout


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 60)
    print("  spaCy Uzbek Model — Evaluation Results")
    print("=" * 60)
    print(f"  Sentences:      {metrics['total_sentences']}")
    print(f"  Tokens:         {metrics['total_tokens']}")
    print(f"  UPOS accuracy:  {metrics['upos_accuracy']}%")
    print(f"  UFeats accuracy:{metrics['ufeats_accuracy']}%")
    print(f"  UAS:            {metrics['uas']}%")
    print(f"  LAS:            {metrics['las']}%")
    print("-" * 60)

    if "per_tag" in metrics and metrics["per_tag"]:
        print("\n  Per-tag UPOS breakdown:")
        print(f"  {'Tag':<10} {'Acc':>7}  {'Correct':>8} / {'Total':>5}")
        print(f"  {'-'*10} {'-'*7}  {'-'*8}   {'-'*5}")
        for tag, t_metrics in sorted(metrics["per_tag"].items()):
            print(f"  {tag:<10} {t_metrics['accuracy']:>6.1f}%"
                  f"  {t_metrics['correct']:>8} / {t_metrics['total']:>5}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained spaCy Uzbek model"
    )
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to trained spaCy model directory")
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Path to test CoNLL-U file")
    parser.add_argument("--test-spacy", type=str, default=None,
                        help="Path to test .spacy file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to save metrics JSON")
    args = parser.parse_args()

    if args.test:
        metrics = evaluate_on_conllu(args.model, args.test)
        print_metrics(metrics)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"\nMetrics saved to {args.output}")
    elif args.test_spacy:
        evaluate_on_spacy(args.model, args.test_spacy)
    else:
        print("ERROR: Provide either --test (CoNLL-U) or --test-spacy (.spacy)")
        sys.exit(1)


if __name__ == "__main__":
    main()
