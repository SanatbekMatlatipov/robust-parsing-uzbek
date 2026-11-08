"""Brute-force verification of decoders.py for n <= 7 (run: python test_decoders.py)."""
import itertools
import json

import numpy as np

from common import save_json
from decoders import (cle_multi_root, cle_one_root, eisner, is_projective, is_valid_tree, matrix_tree, mbr,
                      single_root_laplacian, tree_score)

rng = np.random.default_rng(7)


def random_scores(n):
    s = rng.normal(scale=1.5, size=(n + 1, n + 1))
    np.fill_diagonal(s, -np.inf)
    s[:, 0] = s[:, 0]
    s = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
    s[0, :] = -np.inf
    s[0, 0] = 0.0
    return s


def all_trees(n, single_root=True):
    choices = [[j for j in range(n + 1) if j != i] for i in range(1, n + 1)]
    for hs in itertools.product(*choices):
        heads = np.array((0,) + hs)
        if is_valid_tree(heads, single_root=single_root):
            yield heads


def check(n, trials):
    stats = {"n": n, "trials": trials, "count_ok": 0, "cle_ok": 0, "cle_multi_ok": 0, "eisner_ok": 0,
             "logZ_max_abs_err": 0.0, "mu_max_abs_err": 0.0, "entropy_max_abs_err": 0.0, "mbr_ok": 0}
    for _ in range(trials):
        S = random_scores(n)
        trees = list(all_trees(n))
        stats["count_ok"] += int(len(trees) == n ** (n - 1))
        scores = np.array([tree_score(S, t) for t in trees])
        best = scores.max()
        stats["cle_ok"] += int(abs(tree_score(S, cle_one_root(S)) - best) < 1e-9)
        multi = list(all_trees(n, single_root=False))
        best_multi = max(tree_score(S, t) for t in multi)
        stats["cle_multi_ok"] += int(abs(tree_score(S, cle_multi_root(S)) - best_multi) < 1e-9)
        proj = [t for t in trees if is_projective(t)]
        best_proj = max(tree_score(S, t) for t in proj)
        e = eisner(S)
        stats["eisner_ok"] += int(is_valid_tree(e) and is_projective(e) and abs(tree_score(S, e) - best_proj) < 1e-9)
        # matrix-tree: Z, marginals and entropy against enumeration
        p = np.exp(scores)
        Z = p.sum()
        logZ, mu, H = matrix_tree(S)
        stats["logZ_max_abs_err"] = max(stats["logZ_max_abs_err"], abs(logZ - np.log(Z)))
        q = p / Z
        mu_bf = np.zeros_like(mu)
        for t, w in zip(trees, q):
            for i in range(1, n + 1):
                mu_bf[i, t[i]] += w
        stats["mu_max_abs_err"] = max(stats["mu_max_abs_err"], float(np.abs(mu - mu_bf).max()))
        H_bf = float(-(q * np.log(q)).sum())
        stats["entropy_max_abs_err"] = max(stats["entropy_max_abs_err"], abs(H - H_bf))
        exp_uas = [sum(mu_bf[i, t[i]] for i in range(1, n + 1)) for t in trees]
        stats["mbr_ok"] += int(abs(sum(mu_bf[i, mbr(S)[i]] for i in range(1, n + 1)) - max(exp_uas)) < 1e-9)
    return stats


def unit_weight_counts(nmax=12):
    """det L_hat with unit weights = n^(n-1), the number of single-root trees."""
    out = []
    for n in range(1, nmax + 1):
        theta = np.ones((n + 1, n + 1))
        theta[0, :] = 0.0
        np.fill_diagonal(theta, 0.0)
        det = np.linalg.det(single_root_laplacian(theta))
        out.append({"n": n, "det": round(det), "n^(n-1)": n ** (n - 1)})
    return out


def finite_difference_check():
    """mu = d log Z / d S (gradient identity) checked by central differences."""
    S = random_scores(6)
    _, mu, _ = matrix_tree(S)
    err = 0.0
    eps = 1e-6
    for i in range(1, 7):
        for j in range(0, 7):
            if i == j:
                continue
            Sp, Sm = S.copy(), S.copy()
            Sp[i, j] += eps
            Sm[i, j] -= eps
            g = (matrix_tree(Sp)[0] - matrix_tree(Sm)[0]) / (2 * eps)
            err = max(err, abs(g - mu[i, j]))
    return err


if __name__ == "__main__":
    results = {"brute_force": [check(n, 200 if n <= 5 else 40) for n in range(2, 8)],
               "unit_weight_determinant": unit_weight_counts(),
               "gradient_check_max_abs_err": finite_difference_check()}
    for r in results["brute_force"]:
        print(r)
    print(results["unit_weight_determinant"])
    print("gradient check", results["gradient_check_max_abs_err"])
    ok = all(r["cle_ok"] == r["trials"] and r["cle_multi_ok"] == r["trials"] and r["eisner_ok"] == r["trials"]
             and r["mbr_ok"] == r["trials"] and r["count_ok"] == r["trials"] and r["logZ_max_abs_err"] < 1e-8
             and r["mu_max_abs_err"] < 1e-8 for r in results["brute_force"])
    ok = ok and all(r["det"] == r["n^(n-1)"] for r in results["unit_weight_determinant"])
    results["all_passed"] = bool(ok)
    print("ALL PASSED" if ok else "FAILURES")
    save_json(results, "decoder_tests.json")
