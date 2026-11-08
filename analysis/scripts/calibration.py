"""
Matrix-tree analysis of the trained models (predicted tags, clean input):
  * Z(x) = mass that the locally normalised model gives to well-formed single-root trees,
  * reliability of global arc marginals mu vs local probabilities for the decoded (CLE) arcs, ECE,
  * tree entropy as an error predictor (Spearman correlation with sentence UAS error).
"""
import os

import numpy as np
from scipy import stats

from common import RESULTS, read_conllu, save_json, words

MODELS = ["E1.1", "E2.1", "E3.1", "E5.1", "E1.2", "E2.2", "E3.2"]
TESTS = {"uzudt": "uzudt_test.gold.conllu", "ut": "ut_test.gold.conllu"}
BINS = np.linspace(0, 1, 11)


def ece(conf, correct):
    idx = np.clip(np.digitize(conf, BINS) - 1, 0, 9)
    e = 0.0
    for b in range(10):
        m = idx == b
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return 100 * e


def main():
    out = {"per_model": {}}
    pooled = {"mu": [], "loc": [], "ok": []}
    for m in MODELS:
        out["per_model"][m] = {}
        for t, f in TESTS.items():
            gold = read_conllu(os.path.join(RESULTS, "unified", f))
            z = np.load(os.path.join(RESULTS, "inference", "clean", f"{m}.{t}.pred.npz"), allow_pickle=True)
            mu_c, loc_c, ok, ent, err, Z, n_list = [], [], [], [], [], [], []
            for i, s in enumerate(gold):
                gh = [int(r[6]) for r in words(s)]
                h = z["heads_cle_one_root"][i]
                S, mu = z["scores"][i], z["marginals"][i]
                n = len(gh)
                c = [h[k] == gh[k - 1] for k in range(1, n + 1)]
                mu_c += [mu[k, h[k]] for k in range(1, n + 1)]
                loc_c += [np.exp(S[k, h[k]]) for k in range(1, n + 1)]
                ok += c
                ent.append(z["entropy"][i])
                err.append(1 - np.mean(c))
                Z.append(np.exp(z["logZ"][i]))
                n_list.append(n)
            mu_c, loc_c, ok = np.array(mu_c), np.array(loc_c), np.array(ok, dtype=float)
            rho, p = stats.spearmanr(ent, err)
            rho_n, _ = stats.spearmanr(n_list, err)
            out["per_model"][m][t] = {"ECE_global": ece(mu_c, ok), "ECE_local": ece(loc_c, ok),
                                      "mean_conf_global": float(mu_c.mean()), "mean_conf_local": float(loc_c.mean()),
                                      "accuracy": float(ok.mean()), "mean_Z": float(np.mean(Z)),
                                      "median_Z": float(np.median(Z)),
                                      "spearman_entropy_error": float(rho), "p": float(p),
                                      "spearman_length_error": float(rho_n)}
            pooled["mu"].append(mu_c)
            pooled["loc"].append(loc_c)
            pooled["ok"].append(ok)
    mu, loc, ok = (np.concatenate(pooled[k]) for k in ("mu", "loc", "ok"))
    rel = []
    for b in range(10):
        for name, conf in (("global", mu), ("local", loc)):
            msk = np.clip(np.digitize(conf, BINS) - 1, 0, 9) == b
            if msk.any():
                rel.append((name, b, float(conf[msk].mean()), float(ok[msk].mean()), int(msk.sum())))
    os.makedirs(os.path.join(RESULTS, "csv"), exist_ok=True)
    with open(os.path.join(RESULTS, "csv", "reliability.csv"), "w") as f:
        f.write("kind,bin,confidence,accuracy,count\n")
        for r in rel:
            f.write(",".join(map(str, r)) + "\n")
    out["pooled"] = {"ECE_global": ece(mu, ok), "ECE_local": ece(loc, ok), "arcs": int(len(ok))}
    agg = lambda key, t: float(np.mean([out["per_model"][m][t][key] for m in MODELS]))
    out["mean_over_models"] = {t: {k: agg(k, t) for k in ("ECE_global", "ECE_local", "mean_Z", "median_Z",
                                                          "spearman_entropy_error", "spearman_length_error")}
                               for t in TESTS}
    save_json(out, "calibration.json")
    print(out["pooled"])
    print(out["mean_over_models"])


if __name__ == "__main__":
    main()
