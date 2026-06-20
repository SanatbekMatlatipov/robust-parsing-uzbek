"""
Empirical study for the UzUDT Data-in-Brief reviewer response.

Two experiments using the trained Stanza E2.1 (UzUDT-only +
TahrirchiBERT, last-subword fusion) tagger/parser, invoked via the
same subprocess interface used in the project's training scripts
(`python -m stanza.models.tagger/parser --mode predict ...`):

  A. Robustness perturbation study (Reviewer point 1)
  B. Concrete ABSA parse (Reviewer point 2)

(Reviewer point 5 — UD validate.py warnings — is run separately
against the released CoNLL-U files using the official UD
`tools/validate.py`.)

Run from the workspace root with the project venv active:

    python paper_materials/review_logs/reviewer_empirical_study.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "paper_materials" / "review_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD_TEST = REPO_ROOT / "data" / "depparse" / "uz_uzudt.test.in.conllu"

TAGGER_DIR = REPO_ROOT / "saved_models" / "pos"
TAGGER_NAME = "uz_uzudt_E2.1_tagger.pt"
PARSER_DIR = REPO_ROOT / "saved_models" / "depparse"
PARSER_NAME = "uz_uzudt_E2.1_parser.pt"
BERT_MODEL = "tahrirchi/tahrirchi-bert-base"

ABSA_SENTENCE_TOKENS = [
    "taomlari", "mazali", "emas", ",",
    "lekin", "xizmati", "a'lo", "darajada", ".",
]

EVAL_PY = REPO_ROOT / "scripts" / "eval.py"
spec = importlib.util.spec_from_file_location("ud_eval", EVAL_PY)
ud_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ud_eval)  # type: ignore[union-attr]


# ----------------------------------------------------------------------------
# Perturbations: rewrite only the FORM column. Tree topology is preserved.
# ----------------------------------------------------------------------------

PUNCT = set(".,;:!?\"()[]\u2018\u2019\u201c\u201d\u00ab\u00bb\u2014\u2013-")


def perturb_drop_diacritics(form: str) -> str:
    return (form.replace("\u02bb", "")
                .replace("\u2018", "")
                .replace("\u2019", "")
                .replace("`", ""))


def perturb_lowercase_drop_punct(form: str) -> str:
    if form in PUNCT:
        return ""
    return form.lower()


def perturb_random_caps(form: str) -> str:
    h = sum(ord(c) for c in form)
    if h % 3 == 0:
        return form.upper()
    if h % 3 == 1:
        return form.lower()
    return form.capitalize()


def perturb_codeswitch_ru(form: str) -> str:
    swaps = {
        "va": "и",
        "lekin": "но",
        "ammo": "но",
        "ham": "тоже",
        "juda": "очень",
        "bu": "это",
    }
    h = sum(ord(c) for c in form)
    low = form.lower()
    if low in swaps and h % 2 == 0:
        return swaps[low]
    return form


def perturb_drop_punct_only(form: str) -> str:
    if form in PUNCT:
        return ""
    return form


PERTURBATIONS: "OrderedDict[str, Callable[[str], str]]" = OrderedDict([
    ("clean", lambda s: s),
    ("drop_diacritics", perturb_drop_diacritics),
    ("lowercase_drop_punct", perturb_lowercase_drop_punct),
    ("random_caps", perturb_random_caps),
    ("codeswitch_ru", perturb_codeswitch_ru),
    ("drop_punct_only", perturb_drop_punct_only),
])


# ----------------------------------------------------------------------------
# CoNLL-U rewriting
# ----------------------------------------------------------------------------

def rewrite_forms(in_path: Path, out_path: Path, fn: Callable[[str], str]) -> int:
    out_lines: list[str] = []
    n_touched = 0
    with in_path.open(encoding="utf-8") as f:
        sentence: list[list[str]] = []
        comments: list[str] = []
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if sentence or comments:
                    new_lines, ch = _rebuild_sentence(sentence, comments, fn)
                    out_lines.extend(new_lines)
                    out_lines.append("")
                    n_touched += ch
                sentence, comments = [], []
                continue
            if line.startswith("#"):
                comments.append(line)
                continue
            cols = line.split("\t")
            sentence.append(cols)
        if sentence or comments:
            new_lines, ch = _rebuild_sentence(sentence, comments, fn)
            out_lines.extend(new_lines)
            out_lines.append("")
            n_touched += ch
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return n_touched


def _rebuild_sentence(sentence, comments, fn):
    keep: list[list[str]] = []
    id_map: dict[str, str] = {"0": "0"}
    n_touched = 0
    new_id = 0
    for cols in sentence:
        if len(cols) != 10:
            keep.append(cols)
            continue
        old_id = cols[0]
        if "-" in old_id or "." in old_id:
            keep.append(cols)
            continue
        new_form = fn(cols[1])
        if new_form == "":
            n_touched += 1
            continue
        if new_form != cols[1]:
            n_touched += 1
            cols = list(cols)
            cols[1] = new_form
        new_id += 1
        id_map[old_id] = str(new_id)
        cols = list(cols)
        cols[0] = str(new_id)
        keep.append(cols)
    fixed = []
    seen_root = False
    for cols in keep:
        if len(cols) != 10 or "-" in cols[0] or "." in cols[0]:
            fixed.append(cols)
            continue
        head = cols[6]
        if head not in id_map:
            cols[6] = "0"
            cols[7] = "root"
        else:
            cols[6] = id_map[head]
        if cols[6] == "0":
            if seen_root:
                cols[6] = "1"
                cols[7] = "dep"
            else:
                seen_root = True
                cols[7] = "root"
        fixed.append(cols)
    out = list(comments) + ["\t".join(c) for c in fixed]
    return out, n_touched


# ----------------------------------------------------------------------------
# Subprocess wrappers around stanza.models.{tagger,parser} predict mode
# ----------------------------------------------------------------------------

def run_tagger(eval_file: Path, output_file: Path) -> None:
    cmd = [
        sys.executable, "-m", "stanza.models.tagger",
        "--mode", "predict",
        "--save_dir", str(TAGGER_DIR),
        "--save_name", TAGGER_NAME,
        "--eval_file", str(eval_file),
        "--output_file", str(output_file),
        "--shorthand", "uz_uzudt",
        "--bert_model", BERT_MODEL,
        "--no_pretrain",
        "--lang", "uz",
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def run_parser(eval_file: Path, output_file: Path,
               gold_file: Path | None = None) -> None:
    cmd = [
        sys.executable, "-m", "stanza.models.parser",
        "--mode", "predict",
        "--save_dir", str(PARSER_DIR),
        "--save_name", PARSER_NAME,
        "--eval_file", str(eval_file),
        "--output_file", str(output_file),
        "--shorthand", "uz_uzudt",
        "--bert_model", BERT_MODEL,
        "--no_pretrain",
        "--lang", "uz",
    ]
    if gold_file is not None:
        cmd.extend(["--gold_file", str(gold_file)])
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


# ----------------------------------------------------------------------------
# Eval helper
# ----------------------------------------------------------------------------

def run_eval(gold: Path, pred: Path) -> dict:
    gold_ud = ud_eval.load_conllu_file(str(gold))
    sys_ud = ud_eval.load_conllu_file(str(pred))
    metrics = ud_eval.evaluate(gold_ud, sys_ud)
    return {
        "Tokens": round(metrics["Tokens"].f1 * 100, 2),
        "UPOS": round(metrics["UPOS"].f1 * 100, 2),
        "XPOS": round(metrics["XPOS"].f1 * 100, 2),
        "UFeats": round(metrics["UFeats"].f1 * 100, 2),
        "Lemmas": round(metrics["Lemmas"].f1 * 100, 2),
        "UAS": round(metrics["UAS"].f1 * 100, 2),
        "LAS": round(metrics["LAS"].f1 * 100, 2),
    }


# ----------------------------------------------------------------------------
# Experiment A
# ----------------------------------------------------------------------------

def experiment_A() -> list[dict]:
    rows: list[dict] = []
    for name, fn in PERTURBATIONS.items():
        gold_path = OUT_DIR / f"perturbed_gold_{name}.conllu"
        tagged_path = OUT_DIR / f"perturbed_tagged_{name}.conllu"
        pred_path = OUT_DIR / f"perturbed_pred_{name}.conllu"

        n_touched = rewrite_forms(GOLD_TEST, gold_path, fn)
        run_tagger(gold_path, tagged_path)
        run_parser(tagged_path, pred_path, gold_file=None)
        scores = run_eval(gold_path, pred_path)
        scores["perturbation"] = name
        scores["tokens_modified_or_dropped"] = n_touched
        rows.append(scores)
        print(f"[A] {name:>22s}  touched={n_touched:5d}  "
              f"UPOS={scores['UPOS']:.2f}  UFeats={scores['UFeats']:.2f}  "
              f"UAS={scores['UAS']:.2f}  LAS={scores['LAS']:.2f}")

    out = OUT_DIR / "perturbation_results.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n[A] Saved -> {out}")
    return rows


# ----------------------------------------------------------------------------
# Experiment B
# ----------------------------------------------------------------------------

def experiment_B() -> str:
    sentence_in = OUT_DIR / "absa_input.conllu"
    sentence_tag = OUT_DIR / "absa_tagged.conllu"
    sentence_out = OUT_DIR / "absa_example_parse.conllu"

    lines = ["# sent_id = absa-1",
             "# text = " + " ".join(ABSA_SENTENCE_TOKENS)]
    for i, tok in enumerate(ABSA_SENTENCE_TOKENS, 1):
        lines.append("\t".join([str(i), tok, "_", "_", "_", "_", "0",
                                "dep", "_", "_"]))
    lines.append("")
    sentence_in.write_text("\n".join(lines), encoding="utf-8")

    run_tagger(sentence_in, sentence_tag)
    run_parser(sentence_tag, sentence_out, gold_file=None)

    parsed = sentence_out.read_text(encoding="utf-8")
    print("\n[B] Parsed ABSA sentence:")
    print(parsed)
    return parsed


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["A", "B", "AB"], default="AB")
    cli_args = ap.parse_args()

    if "B" in cli_args.only:
        print("=== Experiment B: ABSA single-sentence parse ===")
        experiment_B()

    if "A" in cli_args.only:
        print("\n=== Experiment A: perturbation study ===")
        rows = experiment_A()
    else:
        rows = []

    summary = {
        "model": {
            "tagger": str(TAGGER_DIR / TAGGER_NAME),
            "parser": str(PARSER_DIR / PARSER_NAME),
            "bert_model": BERT_MODEL,
        },
        "test_set": str(GOLD_TEST),
        "experiment_A_perturbation_table": rows,
        "experiment_B_absa_parse_path": str(OUT_DIR / "absa_example_parse.conllu"),
    }
    out = OUT_DIR / "reviewer_empirical_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nDone. Summary -> {out}")


if __name__ == "__main__":
    main()
