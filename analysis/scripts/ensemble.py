"""
Mixtures of the trained parsers: the arc marginals of the uniform mixture of the
calibrated tree distributions P_b(m) of the models m are the averages of their marginals, and the
minimum-Bayes-risk tree of the mixture (expected number of correct heads) is the single-root maximum spanning
arborescence with these averages as weights. Labels: majority vote of the most probable labels of the models
for the chosen arc (ties broken in the order of the model list).
The model sets are fixed in advance: all seven models, and the three models trained on UzUDT+UT. Both tag
conditions (tags predicted by the paired taggers, gold tags). Paired bootstrap over sentences against the best
single model E2.2 decoded by single-root CLE.
Output: results/ensemble.json
"""
import json
import os
from collections import Counter

import numpy as np

from calibrate import masked
from calibrate_ext import marginals
from common import RESULTS, save_json
from decoders import cle_one_root

ALL = ["E1.1", "E2.1", "E3.1", "E5.1", "E1.2", "E2.2", "E3.2"]
SETS = {"all": ALL, "merged": ["E1.2", "E2.2", "E3.2"]}
TESTS = ("uzudt", "ut")
B = 10000
RNG = np.random.default_rng(11)


def strip(rel):
    return str(rel).split(":")[0]


def per_sentence_scores(models, test, cond, cal):
    """Per sentence: (words, correct heads, correct heads and labels) of the mixture and of E2.2 alone."""
    suffix = ".goldtags" if cond == "gold" else ""
    labs = {m: np.load(os.path.join(RESULTS, "labels", f"{m}.{test}{suffix}.npz"), allow_pickle=True) for m in ALL}
    ref = labs[ALL[0]]
    rows = []
    for s in range(len(ref["gold_heads"])):
        gh, gr = ref["gold_heads"][s], ref["gold_rels"][s]
        n = len(gh) - 1
        mix = None
        for m in models:
            S = masked(labs[m]["scores"][s])
            mu = marginals(cal[m]["beta_global"] * np.where(np.isfinite(S), S, -np.inf))
            mix = mu if mix is None else mix + mu
        mix /= len(models)
        h = cle_one_root(np.where(mix > 0, mix, -1e-9))
        S_best = masked(labs["E2.2"]["scores"][s])
        h_best = cle_one_root(np.where(np.isfinite(S_best), S_best, -1e9))
        u = l = ub = lb = 0
        for i in range(1, n + 1):
            if h[i] == gh[i]:
                u += 1
                votes = Counter(strip(labs[m]["labels"][labs[m]["label_argmax"][s][i, h[i]]]) for m in models)
                best = max(votes.values())
                label = next(strip(labs[m]["labels"][labs[m]["label_argmax"][s][i, h[i]]]) for m in models
                             if votes[strip(labs[m]["labels"][labs[m]["label_argmax"][s][i, h[i]]])] == best)
                l += label == strip(gr[i - 1])
            if h_best[i] == gh[i]:
                ub += 1
                lb += strip(labs["E2.2"]["labels"][labs["E2.2"]["label_argmax"][s][i, h_best[i]]]) == strip(gr[i - 1])
        rows.append((n, u, l, ub, lb))
    return np.array(rows, dtype=float)


def summary(rows):
    n, u, l, ub, lb = rows.T
    out = {"UAS": 100 * u.sum() / n.sum(), "LAS": 100 * l.sum() / n.sum(),
           "UAS_E2.2": 100 * ub.sum() / n.sum(), "LAS_E2.2": 100 * lb.sum() / n.sum()}
    idx = RNG.integers(0, len(rows), size=(B, len(rows)))
    las = 100 * l[idx].sum(1) / n[idx].sum(1)
    diff = las - 100 * lb[idx].sum(1) / n[idx].sum(1)
    out["LAS_ci"] = [float(np.percentile(las, 2.5)), float(np.percentile(las, 97.5))]
    out["diff_vs_E2.2"] = out["LAS"] - out["LAS_E2.2"]
    out["diff_ci"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    out["p_one_sided"] = float(np.mean(diff <= 0))
    return out


def main():
    cal = json.load(open(os.path.join(RESULTS, "calibration_ext.json")))["models"]
    res = {}
    for cond in ("pred", "gold"):
        for name, models in SETS.items():
            for test in TESTS:
                r = summary(per_sentence_scores(models, test, cond, cal))
                res[f"{cond}.{name}.{test}"] = r
                print(cond, name, test, {k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 3)) for k, v in r.items()},
                      flush=True)
    save_json(res, "ensemble.json")


if __name__ == "__main__":
    main()
