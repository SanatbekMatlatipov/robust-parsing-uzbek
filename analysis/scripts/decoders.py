"""
Decoders and matrix-tree computations on an arc-score matrix.

Convention (as in Stanza): S is (n+1) x (n+1), row i = dependent, column j = head, S[i, j] = log P(h_i = j | x);
row 0 (the root symbol) is unused. A tree is an int array `heads` of length n+1 with heads[0] = 0.

  greedy         independent argmax per word (may contain cycles or several root children)
  cle_one_root   Chu-Liu/Edmonds with exactly one root child (Stanza implementation)
  cle_multi_root Chu-Liu/Edmonds without the single-root constraint
  eisner         best projective single-root tree, first-order dynamic programming, O(n^3)
  matrix_tree    single-root matrix-tree theorem (Koo et al., 2007): log Z, arc marginals, entropy
  mbr            minimum-Bayes-risk tree = CLE on the arc marginals (maximises expected UAS)
"""
import importlib.util
import os

import numpy as np

from common import REPO

_spec = importlib.util.spec_from_file_location(
    "chuliu_edmonds", os.path.join(REPO, "stanza", "stanza", "models", "common", "chuliu_edmonds.py"))
_cle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cle)


def greedy(S):
    s = np.array(S, dtype=np.float64)
    np.fill_diagonal(s, -np.inf)
    heads = np.argmax(s, axis=1)
    heads[0] = 0
    return heads


def cle_one_root(S):
    return np.asarray(_cle.chuliu_edmonds_one_root(np.array(S, dtype=np.float64)))


def cle_multi_root(S):
    return np.asarray(_cle.chuliu_edmonds(np.array(S, dtype=np.float64)))


def eisner(S):
    """Best projective tree with exactly one root child. Scores w[h, d] = S[d, h]."""
    S = np.asarray(S, dtype=np.float64)
    n = S.shape[0] - 1
    if n == 1:
        return np.array([0, 0])
    w = S.T  # w[h, d]
    NEG = -np.inf
    # spans over word positions 1..n (index 1..n), arrays indexed [s, t]
    C_l = np.full((n + 2, n + 2), NEG)  # complete, head at t (right end)
    C_r = np.full((n + 2, n + 2), NEG)  # complete, head at s (left end)
    I_l = np.full((n + 2, n + 2), NEG)  # incomplete, arc t -> s
    I_r = np.full((n + 2, n + 2), NEG)  # incomplete, arc s -> t
    bC_l = np.zeros((n + 2, n + 2), dtype=int)
    bC_r = np.zeros((n + 2, n + 2), dtype=int)
    bI = np.zeros((n + 2, n + 2), dtype=int)
    for s in range(1, n + 1):
        C_l[s, s] = C_r[s, s] = 0.0
    for k in range(1, n):
        for s in range(1, n - k + 1):
            t = s + k
            vals = C_r[s, s:t] + C_l[s + 1:t + 1, t]
            r = int(np.argmax(vals))
            best = vals[r]
            I_l[s, t] = best + w[t, s]
            I_r[s, t] = best + w[s, t]
            bI[s, t] = s + r
            vals_l = C_l[s, s:t] + I_l[s:t, t]
            r_l = int(np.argmax(vals_l))
            C_l[s, t] = vals_l[r_l]
            bC_l[s, t] = s + r_l
            vals_r = I_r[s, s + 1:t + 1] + C_r[s + 1:t + 1, t]
            r_r = int(np.argmax(vals_r))
            C_r[s, t] = vals_r[r_r]
            bC_r[s, t] = s + 1 + r_r
    root_scores = np.array([C_l[1, r] + C_r[r, n] + w[0, r] for r in range(1, n + 1)])
    root = 1 + int(np.argmax(root_scores))
    heads = np.zeros(n + 1, dtype=int)
    heads[root] = 0

    def back_complete(s, t, head_left):
        if s == t:
            return
        if head_left:  # C_r[s, t], head s
            r = bC_r[s, t]
            back_incomplete(s, r, head_left=True)
            back_complete(r, t, head_left=True)
        else:  # C_l[s, t], head t
            r = bC_l[s, t]
            back_complete(s, r, head_left=False)
            back_incomplete(r, t, head_left=False)

    def back_incomplete(s, t, head_left):
        if head_left:
            heads[t] = s
        else:
            heads[s] = t
        r = bI[s, t]
        back_complete(s, r, head_left=True)
        back_complete(r + 1, t, head_left=False)

    back_complete(1, root, head_left=False)
    back_complete(root, n, head_left=True)
    return heads


def single_root_laplacian(theta):
    """theta[(n+1),(n+1)] with theta[i, j] = weight of arc j -> i (row = dependent). Returns L_hat (n x n),
    indexed [head-1, dependent-1], first row replaced by the root weights (Koo et al., 2007)."""
    n = theta.shape[0] - 1
    A = theta[1:, 1:].T.copy()  # A[h-1, m-1] = weight of h -> m
    np.fill_diagonal(A, 0.0)
    L = -A
    L[np.diag_indices(n)] = A.sum(axis=0)
    L[0, :] = theta[1:, 0]
    return L


def matrix_tree(S):
    """Single-root matrix-tree theorem on theta = exp(S). Returns log Z, marginals mu (same layout as S:
    mu[m, h] = P(h -> m) under the globally normalised distribution), and the tree entropy."""
    S = np.asarray(S, dtype=np.float64)
    n = S.shape[0] - 1
    shift = 0.0
    theta = np.exp(S - shift)
    theta[0, :] = 0.0
    np.fill_diagonal(theta, 0.0)
    L = single_root_laplacian(theta)
    sign, logdet = np.linalg.slogdet(L)
    if sign <= 0:
        raise ValueError("non-positive determinant")
    Linv = np.linalg.inv(L)
    mu = np.zeros_like(theta)
    for m in range(1, n + 1):
        mu[m, 0] = theta[m, 0] * Linv[m - 1, 0]
        for h in range(1, n + 1):
            if h == m:
                continue
            d = (Linv[m - 1, m - 1] if m != 1 else 0.0) - (Linv[m - 1, h - 1] if h != 1 else 0.0)
            mu[m, h] = theta[m, h] * d
    logZ = logdet
    with np.errstate(divide="ignore", invalid="ignore"):
        logtheta = np.where(theta > 0, np.log(theta), 0.0)
    entropy = logZ - float((mu * logtheta).sum())
    return logZ, mu, entropy


def mbr(S):
    _, mu, _ = matrix_tree(S)
    return cle_one_root(np.where(mu > 0, mu, -1e-9))


def is_valid_tree(heads, single_root=True):
    heads = np.asarray(heads)
    n = len(heads) - 1
    roots = int(np.sum(heads[1:] == 0))
    if roots == 0 or (single_root and roots != 1):
        return False
    for i in range(1, n + 1):
        v, steps = i, 0
        while v != 0 and steps <= n:
            v = heads[v]
            steps += 1
        if v != 0:
            return False
    return True


def is_projective(heads):
    heads = np.asarray(heads)
    n = len(heads) - 1
    for d in range(1, n + 1):
        h = heads[d]
        if h == 0:
            continue
        lo, hi = min(h, d), max(h, d)
        for k in range(lo + 1, hi):
            v = k
            while v != 0 and v != h:
                v = heads[v]
            if v != h:
                return False
    return True


def tree_score(S, heads):
    return float(sum(S[i, heads[i]] for i in range(1, len(heads))))


DECODERS = {"greedy": greedy, "cle_one_root": cle_one_root, "cle_multi_root": cle_multi_root,
            "eisner": eisner, "mbr": mbr}
