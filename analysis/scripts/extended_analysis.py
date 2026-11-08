"""
Extended analysis of the trained parsers.

 (A) Pooled effect per factor with the paired differences summed within every test sentence before the
     bootstrap over the 313 test sentences, so that contrasts evaluated on the same sentences are not treated
     as independent.
 (B) Inverse temperature fitted on the marginal likelihood of the gold heads, sum_i log mu_{g_i i}(beta), instead
     of the joint tree likelihood; calibration criteria of all methods on the same support (the marginal negative
     log-likelihood of the gold heads) and the tree likelihood of the tempered local model restricted to trees.
 (C) Condition-number bound kappa_2(L_beta) <= n^(5/2) / Zbar_beta, where Zbar_beta is the probability that
     independent heads drawn from the tempered local distributions form a tree, checked on all 2191 test matrices
     for beta in {0.05, 0.25, 0.5, 1, 2, 5} together with the lower bound exp(beta Delta / n) / n.
 (D) Centred trace formulas for the variance of the tree weight and the safeguarded Newton iteration with a
     gradient stopping rule and a step-halving rule, compared with the plain iteration of calibrate_ext.py.
 (E) Mixtures: the label of a chosen arc by a vote of the models weighted by their calibrated arc marginals,
     compared with the plurality vote of ensemble.py.
Output: results/extended_analysis.json
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy import optimize

from common import RESULTS, read_conllu, save_json, words
from decoders import cle_one_root, single_root_laplacian, tree_score

MODELS = ["E1.1", "E2.1", "E3.1", "E5.1", "E1.2", "E2.2", "E3.2"]
TESTS = ("uzudt", "ut")
B = 10_000
BETAS = (0.05, 0.25, 0.5, 1.0, 2.0, 5.0)
GOLD = {t: read_conllu(os.path.join(RESULTS, "unified", f"{t}_test.gold.conllu")) for t in TESTS}


# ------------------------------------------------------------------ data
def masked(S):
    S = np.array(S, dtype=np.float64)
    S[0, :] = -np.inf
    np.fill_diagonal(S, -np.inf)
    return S


def gold_heads(gold):
    return [np.array([0] + [int(r[6]) for r in words(s)]) for s in gold]


def test_items(model, test):
    z = np.load(os.path.join(RESULTS, "inference", "clean", f"{model}.{test}.pred.npz"), allow_pickle=True)
    return [masked(s) for s in z["scores"]], [np.asarray(h) for h in z["heads_cle_one_root"]], gold_heads(GOLD[test])


def dev_items(model):
    z = np.load(os.path.join(RESULTS, "calibration_dev", f"{model}.dev.npz"), allow_pickle=True)
    return [(masked(S), np.asarray(g)) for S, g in zip(z["scores"], z["gold"])]


# ------------------------------------------------------------------ matrix-tree machinery
def factors(X):
    valid = np.isfinite(X)
    Xv = np.where(valid, X, -np.inf)
    shift = Xv[1:].max(axis=1)
    psi = np.zeros_like(X)
    psi[1:] = np.exp(Xv[1:] - shift[:, None])
    psi[0] = 0.0
    np.fill_diagonal(psi, 0.0)
    return psi, float(shift.sum())


def inverse_and_marginals(psi):
    L = single_root_laplacian(psi)
    Linv = np.linalg.inv(L)
    n = psi.shape[0] - 1
    mu = np.zeros_like(psi)
    mu[1:, 0] = psi[1:, 0] * Linv[:, 0]
    d = np.diag(Linv).copy()
    d[0] = 0.0
    for m in range(1, n + 1):
        col = Linv[m - 1, :].copy()
        col[0] = 0.0
        mu[m, 1:] = psi[m, 1:] * (d[m - 1] - col)
    np.fill_diagonal(mu, 0.0)
    return L, Linv, mu


def marginals(X):
    return inverse_and_marginals(factors(X)[0])[2]


def log_partition(X):
    psi, shift = factors(X)
    sign, logdet = np.linalg.slogdet(single_root_laplacian(psi))
    assert sign > 0
    return logdet + shift


def arc_features(S):
    n = S.shape[0] - 1
    i = np.arange(n + 1)[:, None]
    j = np.arange(n + 1)[None, :]
    valid = np.isfinite(S)
    w = np.where(valid, S, 0.0)
    nonroot = valid & (j > 0)
    direction = np.where(nonroot & (j > i), 1.0, 0.0)
    length = np.where(nonroot, np.log1p((np.abs(i - j) - 1.0) ** 2), 0.0)
    return w, direction, length


def natural_parameter(S, b=None, phi=None):
    """X = b S (temperature) or X = phi_1 W + phi_2 Dir + phi_3 Len (log-linear family), -inf on invalid arcs."""
    if phi is None:
        return b * S
    w, d, l = arc_features(S)
    return np.where(np.isfinite(S), phi[0] * w + phi[1] * d + phi[2] * l, -np.inf)


def moments(X, feats, center):
    """log Z, means and covariance matrix of additive functionals; with `center` every arc function is replaced by
    f_ji - sum_k mu_ki f_ki (its marginal mean over the heads of the dependent i) before the trace formulas."""
    psi, shift = factors(X)
    L, Linv, mu = inverse_and_marginals(psi)
    sign, logdet = np.linalg.slogdet(L)
    mean = np.array([float((mu * f).sum()) for f in feats])
    if center:
        feats = [f - (mu * f).sum(axis=1, keepdims=True) for f in feats]
    Bm = [Linv @ single_root_laplacian(f * psi) for f in feats]
    K = len(feats)
    cov = np.empty((K, K))
    for k in range(K):
        for l in range(k, K):
            c = np.trace(Linv @ single_root_laplacian(feats[k] * feats[l] * psi)) - np.sum(Bm[k] * Bm[l].T)
            cov[k, l] = cov[l, k] = c
    return logdet + shift, mean, cov


def local_probs(S, b):
    Z = b * S[1:]
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ calibration criteria
def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    e = 0.0
    for k in range(bins):
        sel = idx == k
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return 100 * e


def ece_mass(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    order = np.argsort(conf, kind="stable")
    parts = np.array_split(order, bins)
    return 100 * sum(len(p) / len(conf) * abs(conf[p].mean() - correct[p].mean()) for p in parts if len(p))


def brier(conf, correct):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    return float(np.mean((conf - correct) ** 2))


# ------------------------------------------------------------------ (A) pooled bootstrap by sentence
CONTRASTS = [("pooling", "E2.1", "E3.1"), ("pooling", "E2.2", "E3.2"),
             ("encoder", "E5.1", "E3.1"), ("encoder", "E2.1", "E1.1"), ("encoder", "E2.2", "E1.2"),
             ("data", "E1.2", "E1.1"), ("data", "E2.2", "E2.1"), ("data", "E3.2", "E3.1")]


def sentence_counts(model, test):
    z = np.load(os.path.join(RESULTS, "inference", "clean", f"{model}.{test}.pred.npz"), allow_pickle=True)
    las, tot = [], []
    for i, s in enumerate(GOLD[test]):
        w = words(s)
        h, rel = z["heads_cle_one_root"][i], z["rels_cle_one_root"][i]
        c = sum(1 for k, r in enumerate(w, start=1)
                if h[k] == int(r[6]) and str(rel[k - 1]).split(":")[0] == r[7].split(":")[0])
        las.append(c)
        tot.append(len(w))
    return np.array(las, dtype=float), np.array(tot, dtype=float)


def pooled_by_sentence(rng):
    out = {}
    for fam in ("pooling", "encoder", "data"):
        per_test = {}
        for t in TESTS:
            diff = tot = None
            k = 0
            for f2, a, b in CONTRASTS:
                if f2 != fam:
                    continue
                la, n = sentence_counts(a, t)
                lb, _ = sentence_counts(b, t)
                diff = la - lb if diff is None else diff + la - lb
                tot = n if tot is None else tot + n
                k += 1
            per_test[t] = (diff, tot)
        num = sum(d.sum() for d, _ in per_test.values())
        den = sum(n.sum() for _, n in per_test.values())
        stars, dens = np.zeros(B), np.zeros(B)
        for d, n in per_test.values():
            idx = rng.integers(0, len(n), size=(B, len(n)))
            stars += d[idx].sum(1)
            dens += n[idx].sum(1)
        d_star = 100 * stars / dens
        out[fam] = {"delta_LAS": float(100 * num / den), "ci_low": float(np.percentile(d_star, 2.5)),
                    "ci_high": float(np.percentile(d_star, 97.5)),
                    "p_one_sided": float((np.sum(d_star <= 0) + 1) / (B + 1)),
                    "sentences": int(sum(len(n) for _, n in per_test.values())), "contrasts_per_test": k}
        print("pooled by sentence", fam, out[fam], flush=True)
    return out


# ------------------------------------------------------------------ (B) marginal-likelihood temperature
def marginal_nll(items, b=None, phi=None, local=False):
    total, nw = 0.0, 0
    for S, g in items:
        rows = np.arange(1, len(g))
        if local:
            p = local_probs(S, b)[rows - 1, g[1:]]
        else:
            p = marginals(natural_parameter(S, b, phi))[rows, g[1:]]
        total -= float(np.log(np.maximum(p, 1e-300)).sum())
        nw += len(rows)
    return total, nw


def tree_nll(items, b=None, phi=None):
    total, nw = 0.0, 0
    for S, g in items:
        X = natural_parameter(S, b, phi)
        rows = np.arange(1, len(g))
        total += log_partition(X) - float(X[rows, g[1:]].sum())
        nw += len(rows)
    return total, nw


def fit_marginal(items):
    f = lambda b: marginal_nll(items, b)[0]
    r = optimize.minimize_scalar(f, bounds=(0.05, 5.0), method="bounded", options={"xatol": 1e-6})
    grid = [(b, f(b)) for b in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0)]
    return float(r.x), float(r.fun), grid


def decoded_arcs(S_list, pred, gold, b=None, phi=None, local=False):
    conf, correct, sent = [], [], []
    for s, (S, h, g) in enumerate(zip(S_list, pred, gold)):
        rows = np.arange(1, len(g))
        if local:
            conf += list(local_probs(S, b)[rows - 1, h[1:]])
        else:
            conf += list(marginals(natural_parameter(S, b, phi))[rows, h[1:]])
        correct += list(h[1:] == g[1:])
        sent += [s] * len(rows)
    return np.array(conf), np.array(correct, dtype=float), np.array(sent)


# ------------------------------------------------------------------ (C) condition-number bounds
def contraction_loss(S):
    greedy = sum(S[i, np.isfinite(S[i])].max() for i in range(1, S.shape[0]))
    return float(greedy - tree_score(S, cle_one_root(S)))


def conditioning(S_lists):
    rows = {b: {"kappa": [], "zbar": [], "upper": [], "lower": [], "n": []} for b in BETAS}
    violations = 0
    for S in S_lists:
        n = S.shape[0] - 1
        delta = contraction_loss(S)
        for b in BETAS:
            psi, _ = factors(b * S)
            L = single_root_laplacian(psi)
            kappa = float(np.linalg.cond(L, 2))
            sign, logdet = np.linalg.slogdet(L)
            logzbar = logdet - float(np.log(psi[1:].sum(axis=1)).sum())
            zbar = float(np.exp(logzbar))
            upper = n ** 2.5 / zbar
            lower = np.exp(b * delta / n) / n
            violations += int(not (lower - 1e-9 <= kappa <= upper * (1 + 1e-9)))
            r = rows[b]
            r["kappa"].append(kappa)
            r["zbar"].append(zbar)
            r["upper"].append(upper)
            r["lower"].append(lower)
            r["n"].append(n)
    out = {"matrices": len(S_lists), "violations": violations, "per_beta": {}}
    for b, r in rows.items():
        k, z, u = (np.array(r[x]) for x in ("kappa", "zbar", "upper"))
        out["per_beta"][str(b)] = {"kappa2_max": float(k.max()), "kappa2_median": float(np.median(k)),
                                   "zbar_min": float(z.min()), "zbar_median": float(np.median(z)), "zbar_mean": float(z.mean()),
                                   "upper_max": float(u.max()), "upper_median": float(np.median(u)),
                                   "ratio_upper_over_kappa_min": float((u / k).min()), "ratio_upper_over_kappa_median": float(np.median(u / k)),
                                   "old_upper_median": float(np.median([m ** (1.5 * m) for m in r["n"]]))}
    return out


# ------------------------------------------------------------------ (D) centred variances and safeguarded Newton
def global_obj(items, b, center=True):
    F = g = H = 0.0
    for S, gold in items:
        w = np.where(np.isfinite(S), S, 0.0)
        logZ, mean, cov = moments(b * S, [w], center)
        Wg = float(w[np.arange(1, len(gold)), gold[1:]].sum())
        F += logZ - b * Wg
        g += mean[0] - Wg
        H += cov[0, 0]
    return F, g, H


def newton(obj, items, b0=1.0, lo=0.05, hi=5.0, tol=1e-10, gtol=1e-8, maxit=60):
    """Safeguarded Newton iteration on F'(beta) = 0 for convex F: bracket [lo, hi] with F'(lo) < 0 < F'(hi),
    Newton step accepted when it stays inside the bracket and is at most half the previous step, bisection
    otherwise; stops when |F'| <= gtol or the Newton correction is below tol."""
    while obj(items, lo)[1] >= 0 and lo > 1e-6:
        lo /= 2
    while obj(items, hi)[1] <= 0 and hi < 1e6:
        hi *= 2
    b = min(max(b0, lo), hi)
    prev = hi - lo
    iterates, bisections = [], 0
    for it in range(1, maxit + 1):
        _, g, H = obj(items, b)
        iterates.append(b)
        if g < 0:
            lo = b
        else:
            hi = b
        if abs(g) <= gtol:
            return b, it, iterates, bisections
        step = g / H if H > 0 else np.inf
        if abs(step) < tol:
            return min(max(b - step, lo), hi), it, iterates, bisections
        new = b - step
        if not (lo <= new <= hi) or abs(step) > 0.5 * prev:
            new = 0.5 * (lo + hi)
            bisections += 1
        prev = abs(new - b)
        b = new
    return b, maxit, iterates, bisections


def centring_check(items, betas):
    worst_rel, worst_digits = 0.0, 0.0
    for S, _ in items:
        w = np.where(np.isfinite(S), S, 0.0)
        for b in betas:
            _, mean_u, cov_u = moments(b * S, [w], center=False)
            _, mean_c, cov_c = moments(b * S, [w], center=True)
            rel = abs(cov_u[0, 0] - cov_c[0, 0]) / max(cov_c[0, 0], 1e-300)
            worst_rel = max(worst_rel, rel)
            worst_digits = max(worst_digits, np.log10(max(mean_u[0] ** 2 / max(cov_c[0, 0], 1e-300), 1.0)))
    return {"max_relative_difference": float(worst_rel), "max_log10_mean2_over_var": float(worst_digits)}


# ------------------------------------------------------------------ (E) mixtures with weighted label votes
ALL = MODELS
SETS = {"all": ALL, "merged": ["E1.2", "E2.2", "E3.2"]}


def strip(rel):
    return str(rel).split(":")[0]


def mixture_rows(models, test, cond, betas, rule):
    suffix = ".goldtags" if cond == "gold" else ""
    labs = {m: np.load(os.path.join(RESULTS, "labels", f"{m}.{test}{suffix}.npz"), allow_pickle=True) for m in ALL}
    ref = labs[ALL[0]]
    rows = []
    for s in range(len(ref["gold_heads"])):
        gh, gr = ref["gold_heads"][s], ref["gold_rels"][s]
        n = len(gh) - 1
        mus = {}
        for m in models:
            S = masked(labs[m]["scores"][s])
            mus[m] = marginals(betas[m] * S)
        mix = sum(mus.values()) / len(models)
        h = cle_one_root(np.where(mix > 0, mix, -1e-9))
        S_best = masked(labs["E2.2"]["scores"][s])
        h_best = cle_one_root(np.where(np.isfinite(S_best), S_best, -1e9))
        u = l = ub = lb = 0
        for i in range(1, n + 1):
            if h[i] == gh[i]:
                u += 1
                votes = defaultdict(float)
                order = []
                for m in models:
                    lab = strip(labs[m]["labels"][labs[m]["label_argmax"][s][i, h[i]]])
                    votes[lab] += mus[m][i, h[i]] if rule == "weighted" else 1.0
                    order.append(lab)
                best = max(votes.values())
                label = next(lab for lab in order if votes[lab] == best)
                l += label == strip(gr[i - 1])
            if h_best[i] == gh[i]:
                ub += 1
                lb += strip(labs["E2.2"]["labels"][labs["E2.2"]["label_argmax"][s][i, h_best[i]]]) == strip(gr[i - 1])
        rows.append((n, u, l, ub, lb))
    return np.array(rows, dtype=float)


def mixture_summary(rows, rng):
    n, u, l, ub, lb = rows.T
    out = {"UAS": 100 * u.sum() / n.sum(), "LAS": 100 * l.sum() / n.sum(),
           "UAS_E2.2": 100 * ub.sum() / n.sum(), "LAS_E2.2": 100 * lb.sum() / n.sum()}
    idx = rng.integers(0, len(rows), size=(B, len(rows)))
    las = 100 * l[idx].sum(1) / n[idx].sum(1)
    diff = las - 100 * lb[idx].sum(1) / n[idx].sum(1)
    out["diff_vs_E2.2"] = out["LAS"] - out["LAS_E2.2"]
    out["diff_ci"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    out["p_one_sided"] = float((np.sum(diff <= 0) + 1) / (B + 1))
    return out


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(2026)
    cal = json.load(open(os.path.join(RESULTS, "calibration_ext.json")))["models"]
    out = {}

    # (A)
    out["pooled_by_sentence"] = pooled_by_sentence(rng)

    # (B) and (D)
    out["temperature"] = {"models": {}, "pooled": {}}
    out["newton"] = {}
    pooled = defaultdict(list)
    all_S = []
    for m in MODELS:
        dev = dev_items(m)
        bG, bL, phi = cal[m]["beta_global"], cal[m]["beta_local"], np.array(cal[m]["multi_phi"])
        bM, fM, grid = fit_marginal(dev)
        # (D) safeguarded Newton with centred variances
        b_new, its, iterates, bis = newton(global_obj, dev)
        out["newton"][m] = {"beta_global_json": bG, "beta_global_centred_safeguarded": b_new, "evaluations": its,
                            "bisection_steps": bis, "iterates": iterates, "centring": centring_check(dev, (bG, 5.0))}
        rec = {"beta_global": bG, "beta_local": bL, "beta_marginal": bM, "dev_marginal_nll_per_word": fM / sum(len(g) - 1 for _, g in dev),
               "marginal_nll_grid": grid, "test": {}}
        for t in TESTS:
            S_list, pred, gold = test_items(m, t)
            all_S += S_list
            items = list(zip(S_list, gold))
            nw = sum(len(g) - 1 for g in gold)
            r = {}
            for key, kw in (("global_raw", dict(b=1.0)), ("global_joint", dict(b=bG)), ("global_marginal", dict(b=bM)),
                            ("loglinear", dict(phi=phi)), ("local_raw", dict(b=1.0, local=True)), ("local_cal", dict(b=bL, local=True))):
                conf, corr, sent = decoded_arcs(S_list, pred, gold, **kw)
                r[key] = {"ece": ece(conf, corr), "ece_mass": ece_mass(conf, corr), "brier": brier(conf, corr),
                          "marginal_nll": marginal_nll(items, **kw)[0] / nw}
                if not kw.get("local"):
                    r[key]["tree_nll"] = tree_nll(items, **{k: v for k, v in kw.items() if k != "local"})[0] / nw
                pooled[key].append((t, conf, corr, sent, r[key]["marginal_nll"] * nw, r[key].get("tree_nll", np.nan) * nw, nw))
            r["tree_nll_local_restricted"] = tree_nll(items, b=bL)[0] / nw
            rec["test"][t] = r
        out["temperature"]["models"][m] = rec
        print(m, "beta_G", round(bG, 4), "beta_L", round(bL, 4), "beta_M", round(bM, 4), "newton", round(b_new, 6), its, bis,
              {t: {k: round(v["ece"], 2) for k, v in rec["test"][t].items() if isinstance(v, dict)} for t in TESTS}, flush=True)
    for key, lst in pooled.items():
        conf = np.concatenate([c for _, c, _, _, _, _, _ in lst])
        corr = np.concatenate([c for _, _, c, _, _, _, _ in lst])
        nw = sum(x[6] for x in lst)
        out["temperature"]["pooled"][key] = {"ece": ece(conf, corr), "ece_mass": ece_mass(conf, corr), "brier": brier(conf, corr),
                                             "marginal_nll": sum(x[4] for x in lst) / nw, "arcs": int(len(conf))}
        if not np.isnan(lst[0][5]):
            out["temperature"]["pooled"][key]["tree_nll"] = sum(x[5] for x in lst) / nw
    # bootstrap over test sentences of the pooled ECE of the marginal-likelihood temperature (same scheme as before)
    sizes = {t: len(GOLD[t]) for t in TESTS}
    stats = []
    for _ in range(2000):
        pick = {t: np.bincount(rng.integers(0, sizes[t], size=sizes[t]), minlength=sizes[t]) for t in TESTS}
        conf, corr = [], []
        for t, c, k, sent, _, _, _ in pooled["global_marginal"]:
            wts = pick[t][sent]
            conf.append(np.repeat(c, wts))
            corr.append(np.repeat(k, wts))
        stats.append(ece(np.concatenate(conf), np.concatenate(corr)))
    out["temperature"]["pooled"]["global_marginal"]["ece_test_bootstrap_ci"] = [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]
    print("pooled", {k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)} for k, v in out["temperature"]["pooled"].items()}, flush=True)

    # (C)
    out["conditioning"] = conditioning(all_S)
    print("conditioning", out["conditioning"]["violations"], {b: (round(v["kappa2_max"], 1), round(v["upper_max"], 1), round(v["zbar_min"], 4))
                                                              for b, v in out["conditioning"]["per_beta"].items()}, flush=True)

    # (E)
    betas = {m: cal[m]["beta_global"] for m in MODELS}
    out["mixtures"] = {}
    for rule in ("plurality", "weighted"):
        for cond in ("pred", "gold"):
            for name, models in SETS.items():
                for t in TESTS:
                    r = mixture_summary(mixture_rows(models, t, cond, betas, rule), rng)
                    out["mixtures"][f"{rule}.{cond}.{name}.{t}"] = r
                    print(rule, cond, name, t, {k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 3)) for k, v in r.items()}, flush=True)
    save_json(out, "extended_analysis.json")


if __name__ == "__main__":
    main()
