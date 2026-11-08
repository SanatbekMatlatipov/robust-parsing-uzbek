"""Paired sentence-level bootstrap (B = 10^4) for the controlled contrasts, Holm-Bonferroni adjusted."""
import os

import numpy as np

from common import RESULTS, read_conllu, save_json, words

B = 10_000
RNG = np.random.default_rng(2026)
CONTRASTS = [  # (family, A, B): one-sided H1: A better than B
    ("pooling", "E2.1", "E3.1"), ("pooling", "E2.2", "E3.2"),
    ("encoder", "E5.1", "E3.1"), ("encoder", "E2.1", "E1.1"), ("encoder", "E2.2", "E1.2"),
    ("data", "E1.2", "E1.1"), ("data", "E2.2", "E2.1"), ("data", "E3.2", "E3.1"),
]
TESTS = {"uzudt": "uzudt_test.gold.conllu", "ut": "ut_test.gold.conllu"}


def sentence_counts(model, test, gold):
    z = np.load(os.path.join(RESULTS, "inference", "clean", f"{model}.{test}.pred.npz"), allow_pickle=True)
    las, tot = [], []
    for i, s in enumerate(gold):
        w = words(s)
        h, rel = z["heads_cle_one_root"][i], z["rels_cle_one_root"][i]
        c = sum(1 for k, r in enumerate(w, start=1)
                if h[k] == int(r[6]) and str(rel[k - 1]).split(":")[0] == r[7].split(":")[0])
        las.append(c)
        tot.append(len(w))
    return np.array(las), np.array(tot)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    rows = []
    for test, f in TESTS.items():
        gold = read_conllu(os.path.join(RESULTS, "unified", f))
        cache = {}
        for fam, a, b in CONTRASTS:
            for m in (a, b):
                if m not in cache:
                    cache[m] = sentence_counts(m, test, gold)
            la, tot = cache[a]
            lb, _ = cache[b]
            delta = 100 * (la.sum() - lb.sum()) / tot.sum()
            idx = RNG.integers(0, len(tot), size=(B, len(tot)))
            d_star = 100 * (la[idx].sum(1) - lb[idx].sum(1)) / tot[idx].sum(1)
            rows.append({"test": test, "family": fam, "A": a, "B": b, "delta_LAS": float(delta),
                         "ci_low": float(np.percentile(d_star, 2.5)), "ci_high": float(np.percentile(d_star, 97.5)),
                         "p_one_sided": float((np.sum(d_star <= 0) + 1) / (B + 1))})
    adj = holm(np.array([r["p_one_sided"] for r in rows]))
    for r, p in zip(rows, adj):
        r["p_holm"] = float(p)

    # pooled effect per family: stratified resampling within each (test set, contrast) stratum
    pooled = {}
    golds = {t: read_conllu(os.path.join(RESULTS, "unified", f)) for t, f in TESTS.items()}
    for fam in ("pooling", "encoder", "data"):
        strata = []
        for test in TESTS:
            for f2, a, b in CONTRASTS:
                if f2 == fam:
                    la, tot = sentence_counts(a, test, golds[test])
                    lb, _ = sentence_counts(b, test, golds[test])
                    strata.append((la, lb, tot))
        num = sum(la.sum() - lb.sum() for la, lb, _ in strata)
        den = sum(tot.sum() for _, _, tot in strata)
        stars = np.zeros(B)
        dens = np.zeros(B)
        for la, lb, tot in strata:
            idx = RNG.integers(0, len(tot), size=(B, len(tot)))
            stars += la[idx].sum(1) - lb[idx].sum(1)
            dens += tot[idx].sum(1)
        d_star = 100 * stars / dens
        pooled[fam] = {"delta_LAS": float(100 * num / den), "ci_low": float(np.percentile(d_star, 2.5)),
                       "ci_high": float(np.percentile(d_star, 97.5)),
                       "p_one_sided": float((np.sum(d_star <= 0) + 1) / (B + 1)), "strata": len(strata)}
        print("pooled", fam, pooled[fam])
    save_json({"B": B, "contrasts": rows, "pooled": pooled,
               "note": "one-sided H1: A > B; Holm over all 16 tests; pooled = stratified bootstrap per family"},
              "bootstrap.json")
    for r in rows:
        print(f"{r['test']:6s} {r['family']:8s} {r['A']} vs {r['B']}: dLAS {r['delta_LAS']:+.2f} "
              f"[{r['ci_low']:+.2f}; {r['ci_high']:+.2f}] p={r['p_one_sided']:.4f} holm={r['p_holm']:.4f}")


if __name__ == "__main__":
    main()
