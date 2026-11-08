"""
Expected number of crossing arc pairs of a uniformly random dependency tree conditioned on the total arc length
of the gold tree.

The tilted distribution P_lambda(y) ~ exp(-lambda D(y)) on T(x) is a matrix-tree distribution with the arc factors
exp(-lambda |j - i|) on the non-root arcs and 1 on the root arcs, and P_lambda(y | D(y) = d) = P_U(y | D(y) = d) for
every lambda, because the tilt is a function of D alone. For every gold tree, lambda is chosen so that
E_lambda D = D_obs (bisection; E_lambda D from the arc marginals), trees are sampled exactly from P_lambda head by
head from the conditional marginals (n inversions of the single-root Laplacian per tree, vectorised over samples),
and the samples with D = D_obs estimate E_U[C | D = D_obs]. The sampler is checked against exhaustive enumeration
for n <= 6 and against the uniform laws at lambda = 0.
Output: results/conditional_crossings.json
"""
import itertools
import math
import os
import time
from fractions import Fraction

import numpy as np

from common import RESULTS, save_json
from exact_baselines import crossings, crossings_expectation, orient, pruefer_edges, total_length_moments
from graph_stats import corpus

RNG = np.random.default_rng(2026)
CHUNK = 2000
TARGET_HITS = 200
MAX_SAMPLES = 60000


def tilt_factors(n, lam):
    i = np.arange(n + 1)[:, None]
    j = np.arange(n + 1)[None, :]
    psi = np.exp(-lam * np.abs(i - j).astype(float))
    psi[:, 0] = 1.0
    psi[0, :] = 0.0
    np.fill_diagonal(psi, 0.0)
    return psi


def batched_laplacian(psi):
    """psi: (N, n+1, n+1), psi[k, i, j] = factor of the arc j -> i. Returns L_hat (N, n, n), rows = heads, columns =
    dependents, first row = root factors (Koo et al., 2007)."""
    A = np.transpose(psi[:, 1:, 1:], (0, 2, 1)).copy()      # A[k, h-1, m-1] = psi[k, m, h]
    n = A.shape[1]
    idx = np.arange(n)
    A[:, idx, idx] = 0.0
    L = -A
    L[:, idx, idx] = A.sum(axis=1)
    L[:, 0, :] = psi[:, 1:, 0]
    return L


def head_marginals(psi, Linv, i):
    """Marginal distribution of the head of the word i, (N, n+1), from the inverse Laplacians (N, n, n)."""
    N, n1, _ = psi.shape
    mu = np.zeros((N, n1))
    mu[:, 0] = psi[:, i, 0] * Linv[:, i - 1, 0]
    d = Linv[:, i - 1, i - 1] if i != 1 else np.zeros(N)
    col = Linv[:, i - 1, :].copy()
    col[:, 0] = 0.0
    mu[:, 1:] = psi[:, i, 1:] * (d[:, None] - col)
    mu[:, i] = 0.0
    mu = np.maximum(mu, 0.0)
    return mu / mu.sum(axis=1, keepdims=True)


def sample_trees(n, lam, N):
    """N exact samples from P_lambda on the single-root trees of n words; heads (N, n+1), heads[:, 0] = 0."""
    psi = np.broadcast_to(tilt_factors(n, lam), (N, n + 1, n + 1)).copy()
    heads = np.zeros((N, n + 1), dtype=int)
    for i in range(1, n + 1):
        Linv = np.linalg.inv(batched_laplacian(psi))
        mu = head_marginals(psi, Linv, i)
        u = RNG.random(N)
        h = (np.cumsum(mu, axis=1) < u[:, None]).sum(axis=1)
        h = np.minimum(h, n)
        heads[:, i] = h
        psi[:, i, :] = 0.0
        psi[np.arange(N), i, h] = 1.0
    return heads


def total_length(heads):
    pos = np.arange(heads.shape[1])[None, :]
    d = np.abs(heads - pos)[:, 1:]
    return np.where(heads[:, 1:] == 0, 0, d).sum(axis=1)


def crossings_batched(heads):
    """Crossing pairs of arcs, the root arc drawn from the position 0; heads (N, n+1)."""
    pos = np.arange(heads.shape[1])[None, :]
    lo = np.minimum(heads, pos)[:, 1:]
    hi = np.maximum(heads, pos)[:, 1:]
    a, b = lo[:, :, None], hi[:, :, None]
    c, d = lo[:, None, :], hi[:, None, :]
    cross = (a < c) & (c < b) & (b < d)
    n = lo.shape[1]
    iu = np.triu_indices(n, 1)
    both = cross | np.transpose(cross, (0, 2, 1))
    return both[:, iu[0], iu[1]].sum(axis=1)


def expected_length(n, lam):
    psi = tilt_factors(n, lam)[None]
    Linv = np.linalg.inv(batched_laplacian(psi))
    total = 0.0
    for i in range(1, n + 1):
        mu = head_marginals(psi, Linv, i)[0]
        total += float((mu[1:] * np.abs(np.arange(1, n + 1) - i)).sum())
    return total


def solve_lambda(n, D_obs, lo=-3.0, hi=40.0, iters=80):
    """lambda with E_lambda D = D_obs (E_lambda D decreases in lambda)."""
    f_lo, f_hi = expected_length(n, lo), expected_length(n, hi)
    if not (f_hi < D_obs < f_lo):
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if expected_length(n, mid) > D_obs:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


def conditional_crossings(n, D_obs):
    """Monte Carlo estimate of E_U[C | D = D_obs] with its standard error, and the sample sizes."""
    if n <= 2 or D_obs == n - 1:
        return 0.0, 0.0, 0, 0, None       # all arcs adjacent: no arc can be crossed
    lam = solve_lambda(n, D_obs)
    if lam is None:
        return None, None, 0, 0, None
    hits, s1, s2, total = 0, 0.0, 0.0, 0
    while hits < TARGET_HITS and total < MAX_SAMPLES:
        heads = sample_trees(n, lam, CHUNK)
        total += CHUNK
        sel = total_length(heads) == D_obs
        if sel.any():
            c = crossings_batched(heads[sel]).astype(float)
            hits += int(sel.sum())
            s1 += c.sum()
            s2 += (c * c).sum()
    if hits == 0:
        return None, None, 0, total, lam
    mean = float(s1 / hits)
    var = max(s2 / hits - mean * mean, 0.0)
    return mean, math.sqrt(var / hits), hits, total, lam


def exact_conditional(n):
    """E_U[C | D = d] for every d by enumeration of the n^(n-1) trees."""
    sums, counts = {}, {}
    for seq in itertools.product(range(1, n + 1), repeat=max(n - 2, 0)):
        edges = pruefer_edges(seq, n)
        for r in range(1, n + 1):
            heads = orient(edges, n, r)[1:]
            D = sum(abs(h - d) for d, h in enumerate(heads, start=1) if h != 0)
            sums[D] = sums.get(D, 0) + crossings(heads)
            counts[D] = counts.get(D, 0) + 1
    return {d: Fraction(sums[d], counts[d]) for d in sums}


def checks():
    out = {"enumeration": [], "uniform": []}
    for n in (4, 5, 6):
        exact = exact_conditional(n)
        for d in sorted(exact):
            if d == n - 1 or d == max(exact):
                continue
            est, se, hits, total, lam = conditional_crossings(n, d)
            out["enumeration"].append({"n": n, "D": d, "exact": float(exact[d]), "estimate": est, "se": se, "hits": hits,
                                       "within_3se": bool(est is not None and abs(est - float(exact[d])) <= 3 * max(se, 1e-9) + 1e-9)})
    for n in (5, 8, 12):
        heads = sample_trees(n, 0.0, 20000)
        ED, VD = total_length_moments(n)
        EC = float(sum(crossings_expectation(n)))
        D = total_length(heads)
        C = crossings_batched(heads)
        out["uniform"].append({"n": n, "mean_D": float(D.mean()), "exact_D": float(ED), "se_D": float(D.std() / math.sqrt(len(D))),
                               "var_D": float(D.var()), "exact_var_D": float(VD),
                               "mean_C": float(C.mean()), "exact_C": EC, "se_C": float(C.std() / math.sqrt(len(C)))})
    return out


def main():
    t0 = time.time()
    out = {"checks": checks()}
    print("checks", out["checks"], flush=True)
    trees = corpus("UzUDT") + corpus("UT")
    labels = ["UzUDT"] * len(corpus("UzUDT")) + ["UT"] * len(corpus("UT"))
    rows = []
    for k, ((heads, _), label) in enumerate(zip(trees, labels)):
        n = len(heads)
        if n < 2:
            continue
        D_obs = sum(abs(h - d) for d, h in enumerate(heads, start=1) if h != 0)
        C_obs = crossings(heads)
        ED, VD = total_length_moments(n)
        EC = float(sum(crossings_expectation(n)))
        kappa = (4 * n - 7) / (3 * (3 * n - 4))
        linear = EC + kappa * (D_obs - float(ED))
        est, se, hits, total, lam = conditional_crossings(n, D_obs)
        rows.append({"n": n, "corpus": label, "D": D_obs, "C": C_obs, "EC_U": EC, "linear": linear,
                     "conditional": est, "se": se, "hits": hits, "samples": total, "lambda": lam})
        if k % 100 == 0:
            print(k, rows[-1], round(time.time() - t0, 1), "s", flush=True)
    ok = [r for r in rows if r["conditional"] is not None]
    def summary(sel):
        return {"sentences": len(sel), "observed": float(np.mean([r["C"] for r in sel])),
                "uniform": float(np.mean([r["EC_U"] for r in sel])), "linear": float(np.mean([r["linear"] for r in sel])),
                "conditional": float(np.mean([r["conditional"] for r in sel])),
                "conditional_se": float(math.sqrt(sum(r["se"] ** 2 for r in sel)) / len(sel)),
                "share_linear_negative": float(np.mean([r["linear"] < 0 for r in sel]))}
    bands = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 500)]
    out["summary"] = {"all": summary(ok), "failed": len(rows) - len(ok),
                      "by_corpus": {c: summary([r for r in ok if r["corpus"] == c]) for c in ("UzUDT", "UT")},
                      "by_band": {f"{lo}-{hi}": summary([r for r in ok if lo <= r["n"] <= hi]) for lo, hi in bands
                                  if any(lo <= r["n"] <= hi for r in ok)},
                      "min_hits_positive": int(min(r["hits"] for r in ok if r["hits"] > 0)),
                      "sentences_with_D_min": int(sum(r["hits"] == 0 and r["conditional"] == 0.0 for r in ok)),
                      "seconds": time.time() - t0}
    s = out["summary"]["all"]
    s["share_explained_linear"] = (s["uniform"] - s["linear"]) / (s["uniform"] - s["observed"])
    s["share_explained_conditional"] = (s["uniform"] - s["conditional"]) / (s["uniform"] - s["observed"])
    out["per_sentence"] = rows
    print("summary", out["summary"], flush=True)
    save_json(out, "conditional_crossings.json")


if __name__ == "__main__":
    main()
