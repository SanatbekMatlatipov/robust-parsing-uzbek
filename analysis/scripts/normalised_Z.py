"""
Stanza's inference log-softmax runs over the padded batch length, so exp(S) rows over the real heads sum to
c_i < 1 (training masks the padding). The trained local distribution is the row-renormalised one.
Row scaling multiplies Z by prod_i c_i and leaves the marginals, the entropy and every decoder unchanged,
so only Z(x) and the local probabilities need the correction recorded here.
"""
import os

import numpy as np

from common import RESULTS, read_conllu, save_json, words

MODELS = ["E1.1", "E2.1", "E3.1", "E5.1", "E1.2", "E2.2", "E3.2"]


def main():
    out = {}
    for m in MODELS:
        out[m] = {}
        for t in ("uzudt", "ut"):
            z = np.load(os.path.join(RESULTS, "inference", "clean", f"{m}.{t}.pred.npz"), allow_pickle=True)
            gold = read_conllu(os.path.join(RESULTS, "unified", f"{t}_test.gold.conllu"))
            Z, mass, conf = [], [], []
            for i, s in enumerate(gold):
                S = np.array(z["scores"][i], dtype=np.float64)
                c = np.exp(S[1:]).sum(axis=1)                     # row sums over the real heads
                Z.append(float(np.exp(z["logZ"][i] - np.log(c).sum())))
                mass.append(float(c.mean()))
                h = z["heads_cle_one_root"][i]
                conf += [float(np.exp(S[k, h[k]]) / c[k - 1]) for k in range(1, S.shape[0])]
            out[m][t] = {"mean_Z": float(np.mean(Z)), "median_Z": float(np.median(Z)),
                         "mean_row_mass_on_real_heads": float(np.mean(mass)),
                         "mean_local_conf_decoded": float(np.mean(conf))}
    agg = {t: {"mean_Z": float(np.mean([out[m][t]["mean_Z"] for m in MODELS])),
               "median_Z": float(np.mean([out[m][t]["median_Z"] for m in MODELS])),
               "row_mass": float(np.mean([out[m][t]["mean_row_mass_on_real_heads"] for m in MODELS]))}
           for t in ("uzudt", "ut")}
    save_json({"per_model": out, "mean_over_models": agg}, "normalised_Z.json")
    for m in MODELS:
        print(m, {t: {k: round(v, 3) for k, v in r.items()} for t, r in out[m].items()})
    print(agg)


if __name__ == "__main__":
    main()
