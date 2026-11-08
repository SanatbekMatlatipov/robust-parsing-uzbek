"""
Label distributions of the seven parsers: one more pass over the development sets and the
unified test sets (tags predicted by the paired tagger, clean text) that keeps, besides the arc scores S,
the label scores of the deprel scorer, captured by a forward hook. Stored per sentence:
  scores       S, (n+1) x (n+1), log P(h_i = j | x), as in run_inference.py
  label_argmax most probable label of every arc j -> i, (n+1) x (n+1)  (labels of any decoded tree)
  label_cle    log P(r | x, h_i = j) for the arcs of the single-root CLE tree, n x R
  label_gold   the same for the gold arcs, n x R
Used for labels of minimum-Bayes-risk trees decoded from calibrated marginals, for the calibration of
labelled arcs and for mixtures of the parsers. Output: results/labels/<model>.<split>.npz; with --gold-tags,
the test sets are parsed with gold tags into results/labels/<model>.<split>.goldtags.npz.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from calibrate import clean_copy
from common import RESULTS, read_conllu, words
from decoders import cle_one_root
from run_inference import BATCH, MODELS, UNIFIED, load_parser, load_tagger, tag
from stanza.models.common import utils
from stanza.models.depparse.data import DataLoader as ParseLoader
from stanza.models.depparse.trainer import unpack_batch
from stanza.utils.conll import CoNLL

torch.set_grad_enabled(False)
OUT = os.path.join(RESULTS, "labels")


def parse_with_labels(trainer, pretrain, doc):
    captured = {}
    hook = trainer.model.deprel.register_forward_hook(lambda m, i, o: captured.__setitem__("deprel", o))
    loader = ParseLoader(doc, BATCH, trainer.args, pretrain, vocab=trainer.vocab, evaluation=True,
                         sort_during_eval=True)
    out = []
    for b in loader:
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(b, False)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs
        trainer.model.eval()
        _, preds = trainer.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained,
                                 lemma, head, deprel, word_orig_idx, sentlens, wordlens, sentences=text)
        lab = F.log_softmax(captured["deprel"].float(), dim=3).cpu().numpy()
        batch = [(np.array(preds[0][i][:l, :l], dtype=np.float64), lab[i, :l, :l, :]) for i, l in enumerate(sentlens)]
        out += utils.unsort(batch, orig_idx)
    hook.remove()
    return utils.unsort(out, loader.data_orig_idx)


def run(model, parser, ppt, tagger, tpt, split, gold_path, input_path, gold_tags=False):
    gold = read_conllu(gold_path)
    doc = CoNLL.conll2doc(input_file=input_path) if gold_tags else tag(tagger, tpt, input_path)
    items = parse_with_labels(parser, ppt, doc)
    vocab = parser.vocab["deprel"]
    rec = {k: [] for k in ("scores", "label_argmax", "label_cle", "label_gold", "gold_heads", "gold_rels")}
    for (S, lab), g in zip(items, gold):
        rows = words(g)
        n = len(rows)
        assert S.shape[0] == n + 1
        gh = np.array([0] + [int(r[6]) for r in rows])
        h = np.asarray(cle_one_root(S))
        idx = np.arange(1, n + 1)
        rec["scores"].append(S)
        rec["label_argmax"].append(lab.argmax(axis=2).astype(np.int16))
        rec["label_cle"].append(lab[idx, h[1:], :].astype(np.float32))
        rec["label_gold"].append(lab[idx, gh[1:], :].astype(np.float32))
        rec["gold_heads"].append(gh)
        rec["gold_rels"].append(np.array([r[7] for r in rows]))
    arrays = {}
    for k, v in rec.items():
        a = np.empty(len(v), dtype=object)
        for i, x in enumerate(v):
            a[i] = x
        arrays[k] = a
    labels = np.array([vocab.unmap([i])[0] for i in range(len(vocab))])
    suffix = ".goldtags" if gold_tags else ""
    np.savez_compressed(os.path.join(OUT, f"{model}.{split}{suffix}.npz"), labels=labels, **arrays)
    return len(items)


def main(gold_only=False):
    os.makedirs(OUT, exist_ok=True)
    for name, tagger_file, parser_file, merged, dev_path, _ in MODELS:
        if gold_only:
            parser, ppt, _ = load_parser(parser_file)
            for test in ("uzudt", "ut"):
                path = os.path.join(UNIFIED, f"{test}_test.gold.conllu")
                run(name, parser, ppt, None, None, test, path, path, gold_tags=True)
            print(name, "gold-tag test sets done", flush=True)
            continue
        parser, ppt, _ = load_parser(parser_file)
        tagger, tpt, _ = load_tagger(tagger_file)
        k = run(name, parser, ppt, tagger, tpt, "dev", dev_path, clean_copy(dev_path) if merged else dev_path)
        for test in ("uzudt", "ut"):
            path = os.path.join(UNIFIED, f"{test}_test.gold.conllu")
            k += run(name, parser, ppt, tagger, tpt, test, path, path)
        print(name, "sentences", k, flush=True)


if __name__ == "__main__":
    import sys
    main(gold_only="--gold-tags" in sys.argv)
