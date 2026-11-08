"""
Extended analysis of the temperature calibration.

(1) Exact moments of additive arc functionals under the spanning-tree distribution, by trace formulas
    on the single-root Laplacian Lhat = Lhat[psi] (linear in the arc factors psi):
        E_P F       = tr( Lhat^{-1} Lhat[f psi] ),
        Cov_P(F, G) = tr( Lhat^{-1} Lhat[f g psi] ) - tr( Lhat^{-1} Lhat[f psi] Lhat^{-1} Lhat[g psi] ),
    where F(y) = sum_{a in y} f_a. Checked against enumeration and central differences. They give F' and F''
    of the calibration objective, hence a safeguarded Newton iteration, compared with bisection.
(2) Bootstrap over development sentences: 95% intervals of the fitted inverse temperatures and of the pooled
    test ECE that results from them; bootstrap over test sentences: intervals of the ECE before/after.
(3) Further criteria: equal-mass ECE and the Brier score of the decoded arcs.
(4) Isotonic regression (pool adjacent violators) of the arc marginals: a non-structured reference.
(5) Three-parameter log-linear recalibration P(y) ~ exp(b W(y) + g Dir(y) + l Len(y)), Dir = number of
    head-final arcs, Len = sum of log(1 + (d - 1)^2) over the non-root arcs; convex, fitted by damped Newton.
(6) Diagnostics: share of decoded arcs whose global marginal exceeds the local probability; 1-norm condition
    numbers of Lhat.
Development score matrices are cached in results/calibration_dev/<model>.dev.npz (the first run needs the
checkpoints; later runs need only the cache).
"""
import itertools
import os
import time

import numpy as np

from calibrate import (B_HI, B_LO, OUT, clean_copy, ece, fit, global_nll_grad, gold_heads, local_nll_grad,
                       local_probs, masked, test_items)
from common import RESULTS, save_json
from decoders import cle_one_root, is_valid_tree, single_root_laplacian

RNG = np.random.default_rng(2026)
B_DEV = 1000        # bootstrap resamples of the development sentences
B_TEST_ECE = 200    # of these, the first B_TEST_ECE are propagated to the test ECE
B_TEST = 2000       # bootstrap resamples of the test sentences
TESTS = ("uzudt", "ut")


# ---------------------------------------------------------------- development data (cached)
def dev_items(model):
    name, tagger_file, parser_file, merged, dev_path, _ = model
    path = os.path.join(OUT, f"{name}.dev.npz")
    if not os.path.exists(path):
        from run_inference import load_parser, load_tagger, parse_with_scores, tag
        parser, ppt, _ = load_parser(parser_file)
        tagger, tpt, _ = load_tagger(tagger_file)
        doc = tag(tagger, tpt, clean_copy(dev_path) if merged else dev_path)
        items = parse_with_scores(parser, ppt, doc)
        gold = gold_heads(dev_path)
        scores = np.empty(len(items), dtype=object)
        golds = np.empty(len(items), dtype=object)
        for k, ((S, _, _), g) in enumerate(zip(items, gold)):
            scores[k], golds[k] = S, g
        np.savez_compressed(path, scores=scores, gold=golds)
    z = np.load(path, allow_pickle=True)
    return [(masked(S), np.asarray(g)) for S, g in zip(z["scores"], z["gold"])]


# ---------------------------------------------------------------- features and moments
def arc_features(S):
    """Feature matrices in the layout of S (row = dependent, column = head): W, head-final, log-length kernel."""
    n = S.shape[0] - 1
    i = np.arange(n + 1)[:, None]
    j = np.arange(n + 1)[None, :]
    valid = np.isfinite(S)
    w = np.where(valid, S, 0.0)
    nonroot = valid & (j > 0)
    direction = np.where(nonroot & (j > i), 1.0, 0.0)
    length = np.where(nonroot, np.log1p((np.abs(i - j) - 1.0) ** 2), 0.0)
    return w, direction, length


def factors(X):
    """psi = exp(X) on the valid arcs, each dependent row shifted by its maximum (Z scales by exp(shift))."""
    valid = np.isfinite(X)
    Xv = np.where(valid, X, -np.inf)
    shift = Xv[1:].max(axis=1)
    psi = np.zeros_like(X)
    psi[1:] = np.exp(Xv[1:] - shift[:, None])
    psi[0] = 0.0
    np.fill_diagonal(psi, 0.0)
    return psi, float(shift.sum())


def moments(X, feats):
    """log Z, means and covariance matrix of the additive functionals feats under P(y) ~ prod exp(X_a)."""
    psi, shift = factors(X)
    L = single_root_laplacian(psi)
    sign, logdet = np.linalg.slogdet(L)
    if sign <= 0:
        raise ValueError("non-positive determinant")
    Linv = np.linalg.inv(L)
    B = [Linv @ single_root_laplacian(f * psi) for f in feats]
    mean = np.array([np.trace(Bk) for Bk in B])
    K = len(feats)
    cov = np.empty((K, K))
    for k in range(K):
        for l in range(k, K):
            c = np.trace(Linv @ single_root_laplacian(feats[k] * feats[l] * psi)) - np.sum(B[k] * B[l].T)
            cov[k, l] = cov[l, k] = c
    return logdet + shift, mean, cov, L


def all_trees(n):
    for heads in itertools.product(*[[j for j in range(n + 1) if j != i] for i in range(1, n + 1)]):
        h = np.array((0,) + heads)
        if is_valid_tree(h):
            yield h


def check_moments(trials=60):
    """Trace formulas against enumeration (n <= 6) and F'' against central differences."""
    worst = {"logZ": 0.0, "mean": 0.0, "cov": 0.0, "fd_second": 0.0}
    for _ in range(trials):
        n = int(RNG.integers(2, 7))
        S = RNG.normal(size=(n + 1, n + 1)) * 2.0
        S[0, :] = -np.inf
        np.fill_diagonal(S, -np.inf)
        feats = arc_features(S)
        b = float(RNG.uniform(0.2, 2.0))
        logZ, mean, cov, _ = moments(b * S, feats)
        rows = np.arange(1, n + 1)
        vals, wts = [], []
        for h in all_trees(n):
            v = np.array([f[rows, h[1:]].sum() for f in feats])
            vals.append(v)
            wts.append(b * v[0])
        vals, wts = np.array(vals), np.array(wts)
        lz = np.log(np.exp(wts - wts.max()).sum()) + wts.max()
        p = np.exp(wts - lz)
        m = p @ vals
        c = (vals - m).T @ ((vals - m) * p[:, None])
        worst["logZ"] = max(worst["logZ"], abs(lz - logZ))
        worst["mean"] = max(worst["mean"], np.abs(m - mean).max())
        worst["cov"] = max(worst["cov"], np.abs(c - cov).max())
        h = 1e-5
        d1p = moments((b + h) * S, feats[:1])[1][0]
        d1m = moments((b - h) * S, feats[:1])[1][0]
        worst["fd_second"] = max(worst["fd_second"], abs((d1p - d1m) / (2 * h) - cov[0, 0]))
    return worst


# ---------------------------------------------------------------- one-parameter objectives with F''
def global_obj(items, b):
    F = g = H = 0.0
    for S, gold in items:
        w = arc_features(S)[0]
        logZ, mean, cov, _ = moments(b * np.where(np.isfinite(S), S, -np.inf), [w])
        Wg = float(w[np.arange(1, len(gold)), gold[1:]].sum())
        F += logZ - b * Wg
        g += mean[0] - Wg
        H += cov[0, 0]
    return F, g, H


def local_obj(items, b):
    F = g = H = 0.0
    for S, gold in items:
        P = local_probs(S, b)
        w = np.where(np.isfinite(S[1:]), S[1:], 0.0)
        rows = np.arange(len(gold) - 1)
        m1 = (P * w).sum(axis=1)
        m2 = (P * w * w).sum(axis=1)
        F -= float(np.log(P[rows, gold[1:]]).sum())
        g += float((m1 - w[rows, gold[1:]]).sum())
        H += float((m2 - m1 * m1).sum())
    return F, g, H


def newton(obj, items, b0=1.0, lo=B_LO, hi=B_HI, tol=1e-10, maxit=60):
    """Safeguarded Newton on F' = 0 (F convex). The bracket is first verified, F'(lo) < 0 < F'(hi), and expanded
    if necessary; then the Newton step is taken if it stays in the bracket, a bisection step otherwise. Stops when
    the Newton correction is below tol; returns the root and the number of evaluations of (F', F'')."""
    while obj(items, lo)[1] >= 0 and lo > 1e-6:
        lo /= 2
    while obj(items, hi)[1] <= 0 and hi < 1e6:
        hi *= 2
    b = min(max(b0, lo), hi)
    for it in range(1, maxit + 1):
        _, g, H = obj(items, b)
        if g < 0:
            lo = b
        else:
            hi = b
        step = g / H if H > 0 else np.inf
        if abs(step) < tol:
            return min(max(b - step, lo), hi), it
        new = b - step
        b = new if lo <= new <= hi else 0.5 * (lo + hi)
    return b, maxit


# ---------------------------------------------------------------- calibration criteria
def ece_mass(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    order = np.argsort(conf, kind="stable")
    parts = np.array_split(order, bins)
    return 100 * sum(len(p) / len(conf) * abs(conf[p].mean() - correct[p].mean()) for p in parts if len(p))


def brier(conf, correct):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    return float(np.mean((conf - correct) ** 2))


def pav_fit(conf, correct):
    """Isotonic (non-decreasing) regression of correctness on confidence; returns a predictor."""
    order = np.argsort(conf, kind="stable")
    x, y = np.asarray(conf)[order], np.asarray(correct, dtype=float)[order]
    vals, wts, ends = [], [], []
    for k, v in enumerate(y):
        vals.append(v)
        wts.append(1.0)
        ends.append(k)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2] = (wts[-2] * vals[-2] + wts[-1] * vals[-1]) / w
            wts[-2] = w
            ends[-2] = ends[-1]
            vals.pop(), wts.pop(), ends.pop()
    fitted = np.empty(len(y))
    start = 0
    for v, e in zip(vals, ends):
        fitted[start:e + 1] = v
        start = e + 1
    return lambda c: np.interp(c, x, fitted)


# ---------------------------------------------------------------- per-sentence arc records
def arc_records(S_list, heads_list, gold_list, b_local, b_global, extra=None):
    """Confidence of every decoded arc (local and global), correctness, sentence index; NLL sums."""
    rec = {"local": [], "global": [], "correct": [], "sent": [], "local1": [], "global1": []}
    nll_g = 0.0
    nwords = 0
    for s, (S, h, g) in enumerate(zip(S_list, heads_list, gold_list)):
        n = len(g) - 1
        rows = np.arange(1, n + 1)
        P = local_probs(S, b_local)
        w = arc_features(S)[0]
        X = b_global * np.where(np.isfinite(S), S, -np.inf)
        if extra is not None:
            _, d, l = arc_features(S)
            X = extra[0] * np.where(np.isfinite(S), S, -np.inf) + extra[1] * d + extra[2] * l
            h = cle_one_root(np.where(np.isfinite(X), X, -1e9))
        mu = marginals(X)
        rec["local"] += list(P[rows - 1, h[1:]])
        rec["global"] += list(mu[rows, h[1:]])
        rec["correct"] += list(h[1:] == g[1:])
        rec["sent"] += [s] * n
        logZ = moments(X, [w])[0]
        nll_g += logZ - float(X[rows, g[1:]].sum())
        nwords += n
    out = {k: np.asarray(v) for k, v in rec.items() if v}
    out["nll_global"] = nll_g / nwords
    return out


def marginals(X):
    psi, _ = factors(X)
    L = single_root_laplacian(psi)
    Linv = np.linalg.inv(L)
    n = X.shape[0] - 1
    mu = np.zeros_like(psi)
    mu[1:, 0] = psi[1:, 0] * Linv[:, 0]
    d = np.diag(Linv).copy()
    d[0] = 0.0
    for m in range(1, n + 1):
        col = Linv[m - 1, :].copy()
        col[0] = 0.0
        mu[m, 1:] = psi[m, 1:] * (d[m - 1] - col)
    np.fill_diagonal(mu, 0.0)
    return mu


def multi_obj(items, phi):
    F, g, H = 0.0, np.zeros(3), np.zeros((3, 3))
    for S, gold in items:
        feats = arc_features(S)
        X = sum(p * np.where(np.isfinite(S), f, -np.inf) if k == 0 else p * f for k, (p, f) in enumerate(zip(phi, feats)))
        X = np.where(np.isfinite(S), X, -np.inf)
        logZ, mean, cov, _ = moments(X, list(feats))
        rows = np.arange(1, len(gold))
        Fg = np.array([f[rows, gold[1:]].sum() for f in feats])
        F += logZ - float(phi @ Fg)
        g += mean - Fg
        H += cov
    return F, g, H


def fit_multi(items, phi0, tol=1e-10, maxit=50):
    phi = np.array(phi0, dtype=float)
    F, g, H = multi_obj(items, phi)
    for it in range(1, maxit + 1):
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            new = phi - t * step
            Fn, gn, Hn = multi_obj(items, new)
            if Fn <= F - 1e-4 * t * float(g @ step) or t < 1e-8:
                break
            t *= 0.5
        phi, F, g, H = new, Fn, gn, Hn
        if np.abs(t * step).max() < tol:
            break
    return phi, it


# ---------------------------------------------------------------- main
def main():
    from run_inference import MODELS
    checks = check_moments()
    print("moment checks", checks, flush=True)
    res = {"moment_checks": checks, "models": {}, "pooled": {}}
    pooled = {"raw": [], "cal": [], "iso": [], "multi": []}
    boot_betas = {}
    for model in MODELS:
        name = model[0]
        dev = dev_items(model)
        t0 = time.perf_counter()
        bL_bis, bG_bis = fit(local_nll_grad, dev), fit(global_nll_grad, dev)
        t_bis = time.perf_counter() - t0
        t0 = time.perf_counter()
        bL, itL = newton(local_obj, dev)
        bG, itG = newton(global_obj, dev)
        t_new = time.perf_counter() - t0
        # bootstrap of the development sentences
        bs_L, bs_G = [], []
        for _ in range(B_DEV):
            idx = RNG.integers(0, len(dev), size=len(dev))
            sample = [dev[k] for k in idx]
            bs_L.append(newton(local_obj, sample, b0=bL)[0])
            bs_G.append(newton(global_obj, sample, b0=bG)[0])
        boot_betas[name] = (np.array(bs_L), np.array(bs_G))
        # isotonic reference fitted on the development arcs (global marginals of the CLE trees)
        dev_heads = [cle_one_root(np.where(np.isfinite(S), S, -1e9)) for S, _ in dev]
        dev_rec = arc_records([S for S, _ in dev], dev_heads, [g for _, g in dev], 1.0, 1.0)
        iso = pav_fit(dev_rec["global"], dev_rec["correct"])
        # three-parameter recalibration
        phi, it_multi = fit_multi(dev, [bG, 0.0, 0.0])
        rec = {"beta_local": bL, "beta_global": bG, "beta_local_bisection": bL_bis, "beta_global_bisection": bG_bis,
               "newton_iterations": [itL, itG], "time_bisection_s": t_bis, "time_newton_s": t_new,
               "beta_local_ci": [float(np.percentile(bs_L, 2.5)), float(np.percentile(bs_L, 97.5))],
               "beta_global_ci": [float(np.percentile(bs_G, 2.5)), float(np.percentile(bs_G, 97.5))],
               "multi_phi": phi.tolist(), "multi_iterations": it_multi, "test": {}}
        for test in TESTS:
            S_list, pred, gold = test_items(name, test)
            raw = arc_records(S_list, pred, gold, 1.0, 1.0)
            cal = arc_records(S_list, pred, gold, bL, bG)
            mul = arc_records(S_list, pred, gold, bL, bG, extra=phi)
            iso_conf = iso(raw["global"])
            above = float(np.mean(raw["global"] > raw["local"]))
            conds = []
            for S in S_list:
                for b in (1.0, bG):
                    psi, _ = factors(b * np.where(np.isfinite(S), S, -np.inf))
                    conds.append((b, np.linalg.cond(single_root_laplacian(psi), 1)))
            rec["test"][test] = {
                "ece_global": [ece(raw["global"], raw["correct"])[0], ece(cal["global"], cal["correct"])[0]],
                "ece_local": [ece(raw["local"], raw["correct"])[0], ece(cal["local"], cal["correct"])[0]],
                "ece_mass_global": [ece_mass(raw["global"], raw["correct"]), ece_mass(cal["global"], cal["correct"])],
                "brier_global": [brier(raw["global"], raw["correct"]), brier(cal["global"], cal["correct"])],
                "ece_isotonic": ece(iso_conf, raw["correct"])[0], "brier_isotonic": brier(iso_conf, raw["correct"]),
                "nll_global": [raw["nll_global"], cal["nll_global"], mul["nll_global"]],
                "uas": [float(raw["correct"].mean()), float(mul["correct"].mean())],
                "ece_multi": ece(mul["global"], mul["correct"])[0],
                "share_marginal_above_local": above,
                "cond_max_beta1": float(max(c for b, c in conds if b == 1.0)),
                "cond_max_betaG": float(max(c for b, c in conds if b != 1.0)),
                "cond_median_beta1": float(np.median([c for b, c in conds if b == 1.0])),
            }
            for key, r in (("raw", raw), ("cal", cal), ("multi", mul)):
                pooled[key].append((name, test, r))
            pooled["iso"].append((name, test, {"global": iso_conf, "correct": raw["correct"], "sent": raw["sent"]}))
        res["models"][name] = rec
        print(name, {k: v for k, v in rec.items() if k != "test"}, flush=True)
        for t, r in rec["test"].items():
            print("  ", t, {k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for k, v in r.items()}, flush=True)

    def cat(key, field):
        return np.concatenate([r[field] for _, _, r in pooled[key]])

    for key in ("raw", "cal", "iso", "multi"):
        conf, corr = cat(key, "global"), cat(key, "correct")
        res["pooled"][key] = {"ece": ece(conf, corr)[0], "ece_mass": ece_mass(conf, corr), "brier": brier(conf, corr),
                              "arcs": int(len(conf))}
    for key in ("raw", "cal"):
        conf, corr = cat(key, "local"), cat(key, "correct")
        res["pooled"][key + "_local"] = {"ece": ece(conf, corr)[0], "ece_mass": ece_mass(conf, corr), "brier": brier(conf, corr)}

    # bootstrap over test sentences (the same resampled sentences for every model): pooled global ECE
    sizes = {t: len(test_items("E1.1", t)[2]) for t in TESTS}
    stats = {"raw": [], "cal": [], "diff": []}
    for _ in range(B_TEST):
        pick = {t: np.bincount(RNG.integers(0, sizes[t], size=sizes[t]), minlength=sizes[t]) for t in TESTS}
        vals = {}
        for key in ("raw", "cal"):
            conf, corr = [], []
            for _, t, r in pooled[key]:
                wts = pick[t][r["sent"]]
                conf.append(np.repeat(r["global"], wts))
                corr.append(np.repeat(r["correct"], wts))
            vals[key] = ece(np.concatenate(conf), np.concatenate(corr))[0]
        stats["raw"].append(vals["raw"])
        stats["cal"].append(vals["cal"])
        stats["diff"].append(vals["raw"] - vals["cal"])
    res["pooled"]["test_bootstrap_ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                          for k, v in stats.items()}

    # propagate the development bootstrap of beta to the pooled test ECE
    names = [m[0] for m in MODELS]
    cache = {(name, t): test_items(name, t) for name in names for t in TESTS}
    ece_boot = []
    for b in range(B_TEST_ECE):
        conf, corr = [], []
        for name in names:
            bG = boot_betas[name][1][b]
            for t in TESTS:
                S_list, pred, gold = cache[(name, t)]
                for S, h, g in zip(S_list, pred, gold):
                    mu = marginals(bG * np.where(np.isfinite(S), S, -np.inf))
                    rows = np.arange(1, len(g))
                    conf += list(mu[rows, h[1:]])
                    corr += list(h[1:] == g[1:])
        ece_boot.append(ece(conf, corr)[0])
    res["pooled"]["ece_cal_dev_bootstrap_ci"] = [float(np.percentile(ece_boot, 2.5)), float(np.percentile(ece_boot, 97.5))]
    res["bootstrap"] = {"dev_resamples": B_DEV, "test_resamples": B_TEST, "propagated": B_TEST_ECE}
    save_json(res, "calibration_ext.json")
    print({k: v for k, v in res["pooled"].items()})


if __name__ == "__main__":
    main()
