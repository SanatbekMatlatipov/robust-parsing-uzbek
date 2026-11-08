"""
Numerical checks of the decoding and calibration methods:
 (1) single-pass single-root decoding by root reweighting (Stanojevic and Cohen, 2021): subtracting
     c = n (max w - min w) + 1 from every root arc makes every optimal arborescence single-rooted, so one
     unconstrained run of Chu-Liu/Edmonds solves the constrained problem; compared with root preselection
     (identical trees, timing) and with exhaustive search on random problems;
 (2) conditioning of the scaled single-root Laplacian along the tempered path: the contraction loss
     Delta = sum_i max_j w_ji - max_y W(y), the bounds exp(-b Delta) <= det <= n^(n-1) exp(-b Delta), and the
     1-norm condition numbers for b in [0.05, 5];
 (3) Hessian of the three-parameter log-linear recalibration at its optimum (eigenvalues, condition number);
 (4) minimum-Bayes-risk trees from calibrated marginals (temperature and log-linear) with UAS and LAS;
 (5) multiclass Brier score over all candidate heads of every word;
 (6) calibration of labelled arcs: a label temperature fitted on the development gold arcs, then the
     confidence mu(head) * P(label) of each decoded labelled arc against its correctness;
 (7) the merged-training effect when the merged models read input with the same encoding as their training data.
Output: results/numerics_checks.json
"""
import itertools
import json
import os
import time

import numpy as np

from calibrate import ece, local_probs, masked, test_items
from calibrate_ext import arc_features, dev_items, factors, marginals, multi_obj, newton
from common import RESULTS, save_json
from decoders import _cle, cle_one_root, is_valid_tree, single_root_laplacian, tree_score
from run_inference import MODELS

RNG = np.random.default_rng(7)
TESTS = ("uzudt", "ut")
BETAS = (0.05, 0.25, 0.5, 1.0, 2.0, 5.0)


def strip(rel):
    return str(rel).split(":")[0]


def cle_reweight(S):
    """Single-root optimum by one unconstrained Chu-Liu/Edmonds run on reweighted root arcs."""
    S = np.array(S, dtype=np.float64)
    n = S.shape[0] - 1
    valid = np.isfinite(S)
    valid[0, :] = False
    np.fill_diagonal(valid, False)
    c = n * (S[valid].max() - S[valid].min()) + 1.0
    T = S.copy()
    T[1:, 0] -= c
    return np.asarray(_cle.chuliu_edmonds(T))


def brute_single_root(S):
    n = S.shape[0] - 1
    best, arg = -np.inf, None
    for heads in itertools.product(*[[j for j in range(n + 1) if j != i] for i in range(1, n + 1)]):
        h = np.array((0,) + heads)
        if is_valid_tree(h):
            w = tree_score(S, h)
            if w > best:
                best, arg = w, h
    return arg, best


def check_reweight(trials=300):
    worst = 0.0
    for _ in range(trials):
        n = int(RNG.integers(2, 8))
        S = RNG.normal(size=(n + 1, n + 1)) * 3
        np.fill_diagonal(S, -np.inf)
        h, best = brute_single_root(S)
        r = cle_reweight(S)
        assert is_valid_tree(r)
        worst = max(worst, abs(tree_score(S, r) - best))
    return worst


def contraction_loss(S):
    w = np.where(np.isfinite(S), S, -np.inf)
    greedy = sum(w[i, [j for j in range(w.shape[1]) if j != i]].max() for i in range(1, w.shape[0]))
    h = cle_one_root(S)
    return float(greedy - tree_score(S, h))


def main():
    out = {"reweight": {}, "conditioning": {}, "hessian": {}, "mbr": {}, "brier": {}, "labelled": {}, "matched": {}}
    partial = os.path.join(RESULTS, "numerics_checks.partial.json")
    done = json.load(open(partial)) if os.path.exists(partial) else {}
    out.update({k: v for k, v in done.items() if v})
    cal = json.load(open(os.path.join(RESULTS, "calibration_ext.json")))
    temp = json.load(open(os.path.join(RESULTS, "calibration_temperature.json")))

    # (1) root reweighting
    if not out["reweight"]:
        out["reweight"]["random_problems_max_gap"] = check_reweight()
    same = total = 0
    t_pre, t_rw = [], []
    for name, *_ in (MODELS if "sentences" not in out["reweight"] else []):
        for t in TESTS:
            S_list, pred, _ = test_items(name, t)
            z = np.load(os.path.join(RESULTS, "inference", "clean", f"{name}.{t}.pred.npz"), allow_pickle=True)
            for S in z["scores"]:
                S = np.array(S, dtype=np.float64)
                a = cle_one_root(S)
                b = cle_reweight(S)
                same += bool(np.array_equal(a, b) or abs(tree_score(S, a) - tree_score(S, b)) < 1e-9)
                total += 1
                for fn, acc in ((cle_one_root, t_pre), (cle_reweight, t_rw)):
                    t0 = time.perf_counter()
                    for _ in range(5):
                        fn(S)
                    acc.append((time.perf_counter() - t0) / 5)
    if "sentences" not in out["reweight"]:
        out["reweight"].update({"sentences": total, "identical_optimum": same,
                                "ms_preselection": 1000 * float(np.mean(t_pre)), "ms_reweight": 1000 * float(np.mean(t_rw))})
    print("reweight", out["reweight"], flush=True)
    json.dump(out, open(partial, "w"))

    # (2) conditioning along the tempered path
    deltas, conds, bound_ok = [], {b: [] for b in BETAS}, True
    for name, *_ in (MODELS if not out["conditioning"] else []):
        for t in TESTS:
            S_list, _, _ = test_items(name, t)
            for S in S_list:
                n = S.shape[0] - 1
                delta = contraction_loss(S)
                deltas.append(delta)
                for b in BETAS:
                    X = b * np.where(np.isfinite(S), S, -np.inf)
                    psi, _ = factors(X)
                    L = single_root_laplacian(psi)
                    sign, logdet = np.linalg.slogdet(L)
                    bound_ok &= sign > 0 and (-b * delta - 1e-9 <= logdet <= -b * delta + (n - 1) * np.log(n) + 1e-9)
                    conds[b].append(np.linalg.cond(L, 1))
    deltas = np.array(deltas)
    if not out["conditioning"]:
      out["conditioning"] = {"sentences": int(len(deltas)), "share_delta_zero": float(np.mean(deltas < 1e-12)),
                           "delta_max": float(deltas.max()), "delta_median_positive": float(np.median(deltas[deltas > 1e-12])),
                           "determinant_bounds_hold": bool(bound_ok),
                           "cond_max": {str(b): float(np.max(v)) for b, v in conds.items()},
                           "cond_median": {str(b): float(np.median(v)) for b, v in conds.items()}}
    print("conditioning", out["conditioning"], flush=True)
    json.dump(out, open(partial, "w"))

    # (3) Hessian of the log-linear family at its optimum
    for model in MODELS:
        name = model[0]
        phi = np.array(cal["models"][name]["multi_phi"])
        _, g, H = multi_obj(dev_items(model), phi)
        ev = np.linalg.eigvalsh(H)
        out["hessian"][name] = {"eigenvalues": ev.tolist(), "condition": float(ev[-1] / ev[0]), "grad_norm": float(np.abs(g).max())}
    print("hessian", {k: (round(v["eigenvalues"][0], 2), round(v["condition"], 1)) for k, v in out["hessian"].items()}, flush=True)

    # (4) MBR from calibrated marginals, (5) multiclass Brier, (6) labelled calibration
    def las_uas(heads_list, lab_npz):
        u = l = n_w = 0
        labels = lab_npz["labels"]
        for h, L, gh, gr in zip(heads_list, lab_npz["label_argmax"], lab_npz["gold_heads"], lab_npz["gold_rels"]):
            for i in range(1, len(gh)):
                n_w += 1
                if h[i] == gh[i]:
                    u += 1
                    l += strip(labels[L[i, h[i]]]) == strip(gr[i - 1])
        return 100 * u / n_w, 100 * l / n_w

    mbr_rows = {k: {t: [] for t in TESTS} for k in ("cle", "mbr_1", "mbr_cal", "mbr_loglinear")}
    brier = {k: [] for k in ("local_raw", "local_cal", "global_raw", "global_cal")}
    lab_conf = {"raw": [], "cal": []}
    lab_corr = []
    for model in MODELS:
        name = model[0]
        bG = cal["models"][name]["beta_global"]
        bL = cal["models"][name]["beta_local"]
        phi = np.array(cal["models"][name]["multi_phi"])
        # label temperature on the development gold arcs (convex, Newton)
        dev = np.load(os.path.join(RESULTS, "labels", f"{name}.dev.npz"), allow_pickle=True)
        G = [np.asarray(x, dtype=np.float64) for x in dev["label_gold"]]
        gold_idx = []
        lab_index = {str(v): i for i, v in enumerate(dev["labels"])}
        for gr in dev["gold_rels"]:
            gold_idx.append(np.array([lab_index.get(str(r), lab_index.get(strip(r), 0)) for r in gr]))
        def label_obj(_, bb):
            g1 = h2 = 0.0
            for X, gi in zip(G, gold_idx):
                Z = bb * X
                Z -= Z.max(axis=1, keepdims=True)
                P = np.exp(Z)
                P /= P.sum(axis=1, keepdims=True)
                m1 = (P * X).sum(axis=1)
                m2 = (P * X * X).sum(axis=1)
                g1 += float((m1 - X[np.arange(len(gi)), gi]).sum())
                h2 += float((m2 - m1 * m1).sum())
            return 0.0, g1, h2
        b, _ = newton(label_obj, None)
        out["labelled"].setdefault("beta_label", {})[name] = b
        for t in TESTS:
            lab = np.load(os.path.join(RESULTS, "labels", f"{name}.{t}.npz"), allow_pickle=True)
            S_list, pred, gold = test_items(name, t)
            assert all(np.allclose(np.where(np.isfinite(A), A, 0), np.where(np.isfinite(masked(B)), masked(B), 0))
                       for A, B in zip(S_list, lab["scores"]))
            heads = {"cle": pred, "mbr_1": [], "mbr_cal": [], "mbr_loglinear": []}
            labels = lab["labels"]
            for S, h, g, Lc, gr in zip(S_list, pred, gold, lab["label_cle"], lab["gold_rels"]):
                X1 = np.where(np.isfinite(S), S, -np.inf)
                mu1 = marginals(X1)
                muG = marginals(bG * X1)
                _, d, l = arc_features(S)
                Xphi = phi[0] * X1 + phi[1] * d + phi[2] * l
                muP = marginals(np.where(np.isfinite(S), Xphi, -np.inf))
                for key, mu in (("mbr_1", mu1), ("mbr_cal", muG), ("mbr_loglinear", muP)):
                    heads[key].append(cle_one_root(np.where(mu > 0, mu, -1e-9)))
                n = len(g) - 1
                rows = np.arange(1, n + 1)
                onehot = np.zeros((n, n + 1))
                onehot[rows - 1, g[1:]] = 1.0
                for key, P in (("local_raw", local_probs(S, 1.0)), ("local_cal", local_probs(S, bL)),
                               ("global_raw", mu1[1:]), ("global_cal", muG[1:])):
                    brier[key] += list(((P - onehot) ** 2).sum(axis=1))
                # labelled arcs of the CLE tree
                Lc = np.asarray(Lc, dtype=np.float64)
                rhat = Lc.argmax(axis=1)
                same_rel = np.array([[strip(labels[j]) == strip(labels[rhat[i]]) for j in range(Lc.shape[1])]
                                     for i in range(n)])
                for key, (bh, bl) in (("raw", (1.0, 1.0)), ("cal", (bG, b))):
                    Z = bl * Lc
                    Z -= Z.max(axis=1, keepdims=True)
                    P = np.exp(Z)
                    P /= P.sum(axis=1, keepdims=True)
                    mu = mu1 if key == "raw" else muG
                    lab_conf[key] += list(mu[rows, h[1:]] * (P * same_rel).sum(axis=1))
                lab_corr += [bool(h[i] == g[i] and strip(labels[rhat[i - 1]]) == strip(gr[i - 1])) for i in rows]
            for key, hs in heads.items():
                mbr_rows[key][t].append(las_uas(hs, lab))
    out["mbr"] = {k: {t: {"UAS": float(np.mean([x[0] for x in v[t]])), "LAS": float(np.mean([x[1] for x in v[t]]))}
                      for t in TESTS} for k, v in mbr_rows.items()}
    out["brier"] = {k: float(np.mean(v)) for k, v in brier.items()}
    corr = np.array(lab_corr, dtype=float)
    out["labelled"].update({
        "arcs": int(len(corr)), "las_share": float(corr.mean()),
        "ece_raw": ece(lab_conf["raw"], corr)[0], "ece_cal": ece(lab_conf["cal"], corr)[0],
        "brier_raw": float(np.mean((np.array(lab_conf["raw"]) - corr) ** 2)),
        "brier_cal": float(np.mean((np.array(lab_conf["cal"]) - corr) ** 2)),
        "mean_conf_raw": float(np.mean(lab_conf["raw"])), "mean_conf_cal": float(np.mean(lab_conf["cal"]))})
    print("mbr", out["mbr"], flush=True)
    print("brier", out["brier"], flush=True)
    print("labelled", out["labelled"], flush=True)

    # (7) merged models with matched input encoding
    for name in ("E1.2", "E2.2", "E3.2"):
        r = json.load(open(os.path.join(RESULTS, f"inference_{name}.json")))["results"]
        out["matched"][name] = {t: {"clean": r[f"{t}.clean.pred_tags"]["decoders"]["cle_one_root"]["LAS"],
                                    "matched": r[f"{t}.matched.pred_tags"]["decoders"]["cle_one_root"]["LAS"]}
                                for t in TESTS}
    single = {}
    for name in ("E1.1", "E2.1", "E3.1"):
        r = json.load(open(os.path.join(RESULTS, f"inference_{name}.json")))["results"]
        single[name] = {t: r[f"{t}.clean.pred_tags"]["decoders"]["cle_one_root"]["LAS"] for t in TESTS}
    out["matched"]["merged_effect"] = {
        t: {"clean": float(np.mean([out["matched"][m2][t]["clean"] - single[m1][t] for m1, m2 in
                                     (("E1.1", "E1.2"), ("E2.1", "E2.2"), ("E3.1", "E3.2"))])),
            "matched": float(np.mean([out["matched"][m2][t]["matched"] - single[m1][t] for m1, m2 in
                                       (("E1.1", "E1.2"), ("E2.1", "E2.2"), ("E3.1", "E3.2"))]))} for t in TESTS}
    print("matched", out["matched"], flush=True)
    save_json(out, "numerics_checks.json")


if __name__ == "__main__":
    main()
