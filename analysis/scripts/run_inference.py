"""
Inference with the trained checkpoints (HF Sanatbek/uzudt) on the unified test sets.

For every model: (1) dev sanity check against the training log, (2) tagging with the paired tagger,
(3) parsing with capture of the arc-score matrices S (= Stanza preds[0], log-softmax over heads) and the
best label per arc (preds[1]), (4) all decoders of decoders.py with timing, (5) scoring.

Conditions: gold tags or tags predicted by the paired tagger. The merged (.2) models were trained on
files whose apostrophes are cp1252 mojibake, so they are also run on 'matched' input with the same
corruption applied to FORM and LEMMA (scoring always uses the clean gold file).
"""
import argparse
import importlib.util
import os
import time

import numpy as np
import torch

from stanza.models.common import utils
from stanza.models.common.doc import FEATS, HEAD, DEPREL, UPOS, XPOS
from stanza.models.common.pretrain import Pretrain
from stanza.models.depparse.data import DataLoader as ParseLoader
from stanza.models.depparse.trainer import Trainer as ParseTrainer
from stanza.models.depparse.trainer import unpack_batch
from stanza.models.pos.data import DataLoader as TagLoader
from stanza.models.pos.trainer import Trainer as TagTrainer
from stanza.utils.conll import CoNLL

from common import REPO, RESULTS, cp1252_mojibake, read_conllu, save_json, words, write_conllu
from decoders import DECODERS, is_projective, is_valid_tree, matrix_tree

torch.set_grad_enabled(False)
PT_FILE = os.path.join(REPO, "wordvec", "uz", "pretrain", "fasttext_cc_uz_300.pt")
POS_DIR = os.path.join(REPO, "saved_models", "pos")
DEP_DIR = os.path.join(REPO, "saved_models", "depparse")
OLD_DEV = os.path.join(REPO, "data", "depparse", "uz_uzudt.dev.in.conllu")
MERGED_DEV = os.path.join(REPO, "data", "depparse", "merged", "uz_combined.dev.in.conllu")
UNIFIED = os.path.join(RESULTS, "unified")
BATCH = 5000

# name, tagger, parser, trained on merged (mojibake) data, dev file, best dev LAS in the training log
MODELS = [
    ("E1.1", "uz_uzudt_E1_tagger.pt", "uz_uzudt_E1.1_parser.pt", False, OLD_DEV, 55.28),
    ("E2.1", "uz_uzudt_E2.1_tagger.pt", "uz_uzudt_E2.1_parser.pt", False, OLD_DEV, 58.23),
    ("E3.1", "uz_uzudt_E3.1_tagger.pt", "uz_uzudt_E3.1_parser.pt", False, OLD_DEV, 55.90),
    ("E5.1", "uz_uzudt_E5.1_tagger.pt", "uz_uzudt_E5.1_parser.pt", False, OLD_DEV, 60.56),
    ("E1.2", "uz_combined_E1.2_tagger.pt", "uz_combined_E1.2_parser.pt", True, MERGED_DEV, 62.87),
    ("E2.2", "uz_combined_E2.2_tagger.pt", "uz_combined_E2.2_parser.pt", True, MERGED_DEV, 63.81),
    ("E3.2", "uz_combined_E3.2_tagger.pt", "uz_combined_E3.2_parser.pt", True, MERGED_DEV, None),
]

_spec = importlib.util.spec_from_file_location("ud_eval", os.path.join(REPO, "scripts", "eval.py"))
ud_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ud_eval)


def official_scores(gold_path, sys_path):
    m = ud_eval.evaluate(ud_eval.load_conllu_file(gold_path), ud_eval.load_conllu_file(sys_path))
    return {k: round(m[k].f1 * 100, 2) for k in ("UPOS", "XPOS", "UFeats", "UAS", "LAS", "CLAS", "MLAS", "BLEX")}


_pretrain = None


def pretrain_for(config):
    global _pretrain
    if not config.get("pretrain", False):
        return None
    if _pretrain is None:
        _pretrain = Pretrain(PT_FILE)
    return _pretrain


def load_parser(name):
    path = os.path.join(DEP_DIR, name)
    config = torch.load(path, map_location="cpu", weights_only=False)["config"]
    pt = pretrain_for(config)
    return ParseTrainer(pretrain=pt, model_file=path, use_cuda=False), pt, config


def load_tagger(name):
    path = os.path.join(POS_DIR, name)
    config = torch.load(path, map_location="cpu", weights_only=False)["config"]
    pt = pretrain_for(config)
    return TagTrainer(pretrain=pt, model_file=path, use_cuda=False), pt, config


def tag(trainer, pretrain, conllu_path):
    doc = CoNLL.conll2doc(input_file=conllu_path)
    loader = TagLoader(doc, BATCH, trainer.args, pretrain, vocab=trainer.vocab, evaluation=True,
                       sort_during_eval=True)
    preds = []
    for b in loader:
        preds += trainer.predict(b)
    preds = utils.unsort(preds, loader.data_orig_idx)
    loader.doc.set([UPOS, XPOS, FEATS], [y for x in preds for y in x])
    return loader.doc


def parse_with_scores(trainer, pretrain, doc):
    """Returns, in document order, a list of (S, label_index_matrix, n) per sentence."""
    loader = ParseLoader(doc, BATCH, trainer.args, pretrain, vocab=trainer.vocab, evaluation=True,
                         sort_during_eval=True)
    out = []
    for b in loader:
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(b, False)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs
        trainer.model.eval()
        _, preds = trainer.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained,
                                 lemma, head, deprel, word_orig_idx, sentlens, wordlens, sentences=text)
        batch_items = [(np.array(preds[0][i][:l, :l], dtype=np.float64), np.array(preds[1][i][:l, :l]), l - 1)
                       for i, l in enumerate(sentlens)]
        out += utils.unsort(batch_items, orig_idx)
    return utils.unsort(out, loader.data_orig_idx)


def strip(rel):
    return rel.split(":")[0]


def score_heads(gold_sents, systems, field):
    """Simple word-level UAS/LAS (gold tokenisation, deprel subtypes stripped as in the CoNLL 2018 scorer)."""
    total = uas = las = 0
    for g, s in zip(gold_sents, systems):
        for i, r in enumerate(words(g), start=1):
            total += 1
            if s["heads"][i] == int(r[6]):
                uas += 1
                if strip(s[field][i - 1]) == strip(r[7]):
                    las += 1
    return 100 * uas / total, 100 * las / total


def tag_accuracy(gold_sents, doc):
    pred_rows = [[w.upos or "_", w.xpos or "_", w.feats or "_"] for s in doc.sentences for w in s.words]
    gold_rows = [(r[3], r[4], r[5]) for g in gold_sents for r in words(g)]
    n = len(gold_rows)
    return {"UPOS": 100 * sum(p[0] == q[0] for p, q in zip(pred_rows, gold_rows)) / n,
            "XPOS": 100 * sum(p[1] == q[1] for p, q in zip(pred_rows, gold_rows)) / n,
            "UFeats": 100 * sum(p[2] == q[2] for p, q in zip(pred_rows, gold_rows)) / n}


def write_system(gold_sents, heads_rels, path):
    out = []
    for g, (heads, rels) in zip(gold_sents, heads_rels):
        rows = []
        k = 0
        for r in g["rows"]:
            r = list(r)
            if r[0].isdigit():
                k += 1
                r[6], r[7] = str(int(heads[k])), rels[k - 1]
                r[8] = "_"
            rows.append(r)
        out.append({"comments": g["comments"], "rows": rows})
    write_conllu(out, path)


def run_condition(model, parser, ppt, tagger, tpt, test_name, gold_path, input_path, use_pred_tags, out_dir):
    gold_sents = read_conllu(gold_path)
    t0 = time.perf_counter()
    doc = tag(tagger, tpt, input_path) if use_pred_tags else CoNLL.conll2doc(input_file=input_path)
    tag_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    items = parse_with_scores(parser, ppt, doc)
    score_time = time.perf_counter() - t0
    deprel_vocab = parser.vocab["deprel"]
    result = {"sentences": len(items), "words": int(sum(n for _, _, n in items)),
              "tagging_time_s": tag_time, "scoring_time_s": score_time, "decoders": {}}
    if use_pred_tags:
        result["tagging"] = tag_accuracy(gold_sents, doc)
    systems = {k: [] for k in DECODERS}
    times = {k: [] for k in DECODERS}
    mt = []
    for S, L, n in items:
        for name, fn in DECODERS.items():
            t0 = time.perf_counter()
            heads = fn(S)
            times[name].append((n, time.perf_counter() - t0))
            rels = [deprel_vocab.unmap([int(L[i, heads[i]])])[0] for i in range(1, n + 1)]
            systems[name].append({"heads": heads, "rels": rels, "valid": is_valid_tree(heads),
                                  "projective": bool(is_valid_tree(heads) and is_projective(heads))})
        logZ, mu, H = matrix_tree(S)
        mt.append((n, logZ, H, mu))
    for name in DECODERS:
        uas, las = score_heads(gold_sents, systems[name], "rels")
        result["decoders"][name] = {
            "UAS": uas, "LAS": las,
            "valid_tree_share": 100 * np.mean([s["valid"] for s in systems[name]]),
            "projective_share": 100 * np.mean([s["projective"] for s in systems[name]]),
            "ms_per_sentence": 1000 * float(np.mean([t for _, t in times[name]])),
        }
    sys_path = os.path.join(out_dir, f"{model}.{test_name}.{'pred' if use_pred_tags else 'gold'}.cle.conllu")
    write_system(gold_sents, [(s["heads"], s["rels"]) for s in systems["cle_one_root"]], sys_path)
    result["official_cle_one_root"] = official_scores(gold_path, sys_path)
    if use_pred_tags:
        # official MLAS/UPOS need predicted tags in the file: add them
        tagged = [[w.upos or "_", w.xpos or "_", w.feats or "_"] for s in doc.sentences for w in s.words]
        sents = read_conllu(sys_path)
        k = 0
        for s in sents:
            for r in s["rows"]:
                if r[0].isdigit():
                    r[3], r[4], r[5] = tagged[k]
                    k += 1
        write_conllu(sents, sys_path)
        result["official_cle_one_root"] = official_scores(gold_path, sys_path)
    # per-sentence records for bootstrap, calibration and timing
    np.savez_compressed(
        os.path.join(out_dir, f"{model}.{test_name}.{'pred' if use_pred_tags else 'gold'}.npz"),
        n=np.array([n for _, _, n in items]),
        logZ=np.array([m[1] for m in mt]), entropy=np.array([m[2] for m in mt]),
        **{f"heads_{k}": np.array([np.array(s["heads"]) for s in systems[k]], dtype=object) for k in DECODERS},
        **{f"rels_{k}": np.array([np.array(s["rels"]) for s in systems[k]], dtype=object) for k in DECODERS},
        **{f"time_{k}": np.array([t for _, t in times[k]]) for k in DECODERS},
        scores=np.array([S for S, _, _ in items], dtype=object),
        marginals=np.array([m[3] for m in mt], dtype=object),
    )
    return result


def dev_check(parser, ppt, dev_path, out_dir, model):
    doc = CoNLL.conll2doc(input_file=dev_path)
    items = parse_with_scores(parser, ppt, doc)
    deprel_vocab = parser.vocab["deprel"]
    gold = read_conllu(dev_path)
    pairs = []
    for S, L, n in items:
        heads = DECODERS["cle_one_root"](S)
        pairs.append((heads, [deprel_vocab.unmap([int(L[i, heads[i]])])[0] for i in range(1, n + 1)]))
    path = os.path.join(out_dir, f"{model}.dev.cle.conllu")
    write_system(gold, pairs, path)
    return official_scores(dev_path, path)


def mojibake_copy(path):
    out = path.replace(".gold.conllu", ".gold.mojibake.conllu")
    sents = read_conllu(path)
    for s in sents:
        for r in s["rows"]:
            r[1] = cp1252_mojibake(r[1])
            r[2] = cp1252_mojibake(r[2])
    write_conllu(sents, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[m[0] for m in MODELS])
    args = ap.parse_args()
    out_dir = os.path.join(RESULTS, "inference")
    os.makedirs(out_dir, exist_ok=True)
    tests = {"uzudt": os.path.join(UNIFIED, "uzudt_test.gold.conllu"), "ut": os.path.join(UNIFIED, "ut_test.gold.conllu")}
    moji = {k: mojibake_copy(v) for k, v in tests.items()}
    for name, tagger_file, parser_file, merged, dev_path, logged in MODELS:
        if name not in args.models:
            continue
        t0 = time.perf_counter()
        parser, ppt, pconf = load_parser(parser_file)
        tagger, tpt, tconf = load_tagger(tagger_file)
        record = {"model": name, "tagger": tagger_file, "parser": parser_file,
                  "parser_config": {k: pconf.get(k) for k in ("bert_model", "bert_pooling", "pretrain", "train_file")},
                  "tagger_config": {k: tconf.get(k) for k in ("bert_model", "bert_pooling", "pretrain", "train_file")},
                  "load_time_s": time.perf_counter() - t0}
        record["dev_check"] = dev_check(parser, ppt, dev_path, out_dir, name)
        record["dev_logged_best_LAS"] = logged
        print(name, "dev", record["dev_check"]["LAS"], "logged", logged, flush=True)
        record["results"] = {}
        for test_name, gold_path in tests.items():
            inputs = {"clean": gold_path}
            if merged:
                inputs["matched"] = moji[test_name]
            for variant, input_path in inputs.items():
                for use_pred in (False, True):
                    key = f"{test_name}.{variant}.{'pred' if use_pred else 'gold'}_tags"
                    sub = os.path.join(out_dir, variant)
                    os.makedirs(sub, exist_ok=True)
                    record["results"][key] = run_condition(name, parser, ppt, tagger, tpt, test_name, gold_path,
                                                           input_path, use_pred, sub)
                    r = record["results"][key]
                    print(name, key, {k: round(v["LAS"], 2) for k, v in r["decoders"].items()},
                          "official", r["official_cle_one_root"]["LAS"], flush=True)
        save_json(record, f"inference_{name}.json")


if __name__ == "__main__":
    main()
