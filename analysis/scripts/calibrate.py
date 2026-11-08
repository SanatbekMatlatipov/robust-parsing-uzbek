"""
Temperature calibration of the local head distributions and of the matrix-tree (global) distribution.

Local:  p_b(j | i) = exp(b S[i, j]) / sum_j' exp(b S[i, j'])            (S = Stanza log-softmax head scores)
Global: P_b(y | x) = exp(b W(y)) / Z_b(x),  W(y) = sum_{(h -> m) in y} S[m, h],  Z_b = det L_hat(exp(b S))

Both negative log-likelihoods are convex in the inverse temperature b; their derivatives are
  local:  sum_i ( E_{p_b}[S[i, .]] - S[i, g_i] ),
  global: sum_x ( sum_a mu_b(a) S(a) - W(y*) ),
non-decreasing in b, so b is found by bisection on the derivative. b is fitted on the development set of
each model (tags predicted by the paired tagger, clean text), then applied to the unified test sets.
The decoded trees do not change: argmax_y b W(y) = argmax_y W(y) for every b > 0.
"""
import os

import numpy as np

from common import RESULTS, read_conllu, save_json, words, write_conllu
from decoders import matrix_tree
from run_inference import MODELS, load_parser, load_tagger, parse_with_scores, tag

B_LO, B_HI, ITERS = 0.05, 5.0, 50
N_BINS = 10
OUT = os.path.join(RESULTS, "calibration_dev")


def cp1252_byte(c):
    try:
        return c.encode("cp1252")[0]
    except UnicodeEncodeError:
        if ord(c) < 256:
            return ord(c)
        raise


def repair_mojibake(text):
    """Inverse of common.cp1252_mojibake; tokens that are not mojibake are returned unchanged."""
    if all(ord(c) < 128 for c in text):
        return text
    try:
        return bytes(cp1252_byte(c) for c in text).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError, ValueError):
        return text


def clean_copy(path):
    sents = read_conllu(path)
    for s in sents:
        s["comments"] = [repair_mojibake(c) for c in s["comments"]]
        for r in s["rows"]:
            r[1], r[2] = repair_mojibake(r[1]), repair_mojibake(r[2])
    out = os.path.join(OUT, os.path.basename(path).replace(".conllu", ".clean.conllu"))
    write_conllu(sents, out)
    return out


def gold_heads(path):
    return [np.array([0] + [int(r[6]) for r in words(s)]) for s in read_conllu(path)]


def masked(S):
    S = np.array(S, dtype=np.float64)
    S[0, :] = -np.inf
    np.fill_diagonal(S, -np.inf)
    return S


def local_probs(S, b):
    Z = b * S[1:]
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)          # rows = words 1..n, columns = heads 0..n


def local_nll_grad(items, b):
    nll = grad = 0.0
    for S, g in items:
        P = local_probs(S, b)
        rows = np.arange(1, len(g))
        Sg = S[rows, g[1:]]
        finite = np.where(np.isfinite(S[1:]), S[1:], 0.0)
        logp = np.log(P[np.arange(len(rows)), g[1:]])
        nll -= logp.sum()
        grad += float(((P * finite).sum(axis=1) - Sg).sum())
    return nll, grad


def global_nll_grad(items, b):
    nll = grad = 0.0
    for S, g in items:
        finite = np.where(np.isfinite(S), S, -1e9)
        logZ, mu, _ = matrix_tree(b * finite)
        W = float(S[np.arange(1, len(g)), g[1:]].sum())
        nll += logZ - b * W
        grad += float((mu * np.where(np.isfinite(S), S, 0.0)).sum()) - W
    return nll, grad


def fit(fn, items):
    lo, hi = B_LO, B_HI
    if fn(items, hi)[1] < 0:
        return hi
    for _ in range(ITERS):
        mid = 0.5 * (lo + hi)
        if fn(items, mid)[1] < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ece(conf, correct, bins=N_BINS):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    e, table = 0.0, []
    for k in range(bins):
        sel = idx == k
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
            table.append({"bin": k, "count": int(sel.sum()), "confidence": float(conf[sel].mean()),
                          "accuracy": float(correct[sel].mean())})
    return 100 * e, table


def test_items(model, test):
    z = np.load(os.path.join(RESULTS, "inference", "clean", f"{model}.{test}.pred.npz"), allow_pickle=True)
    gold = gold_heads(os.path.join(RESULTS, "unified", f"{test}_test.gold.conllu"))
    S = [masked(s) for s in z["scores"]]
    pred = [np.asarray(h) for h in z["heads_cle_one_root"]]
    return S, pred, gold


def evaluate(S_list, pred, gold, b_local, b_global):
    rec = {"local": {"conf": [], "correct": []}, "global": {"conf": [], "correct": []}}
    nll_local = nll_global = words_n = 0.0
    for S, h, g in zip(S_list, pred, gold):
        n = len(g) - 1
        rows = np.arange(1, n + 1)
        P = local_probs(S, b_local)
        rec["local"]["conf"] += list(P[rows - 1, h[1:]])
        nll_local -= float(np.log(P[rows - 1, g[1:]]).sum())
        finite = np.where(np.isfinite(S), S, -1e9)
        logZ, mu, _ = matrix_tree(b_global * finite)
        rec["global"]["conf"] += list(mu[rows, h[1:]])
        nll_global += logZ - b_global * float(S[rows, g[1:]].sum())
        correct = list(h[1:] == g[1:])
        rec["local"]["correct"] += correct
        rec["global"]["correct"] += correct
        words_n += n
    return rec, nll_local / words_n, nll_global / words_n


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {"models": {}, "pooled": {}}
    pooled = {k: {"conf": [], "correct": []} for k in ("local_raw", "global_raw", "local_cal", "global_cal")}
    for name, tagger_file, parser_file, merged, dev_path, _ in MODELS:
        parser, ppt, _ = load_parser(parser_file)
        tagger, tpt, _ = load_tagger(tagger_file)
        dev_input = clean_copy(dev_path) if merged else dev_path
        doc = tag(tagger, tpt, dev_input)
        items = parse_with_scores(parser, ppt, doc)
        gold = gold_heads(dev_path)
        dev = [(masked(S), g) for (S, _, _), g in zip(items, gold)]
        assert all(S.shape[0] == len(g) for S, g in dev)
        b_l, b_g = fit(local_nll_grad, dev), fit(global_nll_grad, dev)
        dev_words = sum(len(g) - 1 for _, g in dev)
        rec = {"beta_local": b_l, "beta_global": b_g, "dev_sentences": len(dev), "dev_words": dev_words,
               "dev_nll_local": {"raw": local_nll_grad(dev, 1.0)[0] / dev_words, "cal": local_nll_grad(dev, b_l)[0] / dev_words},
               "dev_nll_global": {"raw": global_nll_grad(dev, 1.0)[0] / dev_words, "cal": global_nll_grad(dev, b_g)[0] / dev_words},
               "test": {}}
        for test in ("uzudt", "ut"):
            S_list, pred, gold_t = test_items(name, test)
            raw, nl_raw, ng_raw = evaluate(S_list, pred, gold_t, 1.0, 1.0)
            cal, nl_cal, ng_cal = evaluate(S_list, pred, gold_t, b_l, b_g)
            rec["test"][test] = {
                "ece_local_raw": ece(raw["local"]["conf"], raw["local"]["correct"])[0],
                "ece_local_cal": ece(cal["local"]["conf"], cal["local"]["correct"])[0],
                "ece_global_raw": ece(raw["global"]["conf"], raw["global"]["correct"])[0],
                "ece_global_cal": ece(cal["global"]["conf"], cal["global"]["correct"])[0],
                "nll_local_raw": nl_raw, "nll_local_cal": nl_cal, "nll_global_raw": ng_raw, "nll_global_cal": ng_cal,
                "mean_conf_local_cal": float(np.mean(cal["local"]["conf"])),
                "mean_conf_global_cal": float(np.mean(cal["global"]["conf"])),
                "accuracy": float(np.mean(raw["local"]["correct"])),
            }
            for kind in ("local", "global"):
                pooled[f"{kind}_raw"]["conf"] += raw[kind]["conf"]
                pooled[f"{kind}_raw"]["correct"] += raw[kind]["correct"]
                pooled[f"{kind}_cal"]["conf"] += cal[kind]["conf"]
                pooled[f"{kind}_cal"]["correct"] += cal[kind]["correct"]
        results["models"][name] = rec
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in rec.items() if k != "test"},
              {t: {k: round(v, 2) for k, v in r.items()} for t, r in rec["test"].items()}, flush=True)
    rows = []
    for key, d in pooled.items():
        e, table = ece(d["conf"], d["correct"])
        results["pooled"][key] = {"ece": e, "arcs": len(d["conf"]), "bins": table}
        rows += [(key, r["confidence"], r["accuracy"], r["count"]) for r in table]
    save_json(results, "calibration_temperature.json")
    with open(os.path.join(RESULTS, "csv", "reliability_temperature.csv"), "w") as f:
        f.write("kind,confidence,accuracy,count\n")
        for kind, c, a, n in rows:
            f.write(f"{kind},{c:.5f},{a:.5f},{n}\n")
    print({k: round(v["ece"], 2) for k, v in results["pooled"].items()})


if __name__ == "__main__":
    main()
