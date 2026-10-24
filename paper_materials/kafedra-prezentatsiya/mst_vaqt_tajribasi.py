"""
Decoding-time experiment for the seminar slides (slides 6, 12-14):
exhaustive search over all dependency trees vs. Chu-Liu-Edmonds (CLE).

Uses Stanza's own CLE implementation from this repository
(stanza/stanza/models/common/chuliu_edmonds.py), which needs only NumPy.

  1. tree count:  the number of single-root trees equals n^(n-1)  (n <= 7)
  2. correctness: CLE (one root) finds the exhaustive optimum       (n <= 7)
  3. time:        exhaustive search for n = 2..8, CLE for n = 2..150
  4. test set:    CLE on the real UzUDT test sentence lengths

Weights are random log-softmax rows: the greedy step then produces many
cycles, so this is a hard case for CLE.

Usage (from the repository root):
  python paper_materials/kafedra-prezentatsiya/mst_vaqt_tajribasi.py
"""
import importlib.util
import itertools
import json
import os
import platform
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "mst_vaqt_natijalari.json")

spec = importlib.util.spec_from_file_location(
    "cle", os.path.join(REPO, "stanza", "stanza", "models", "common", "chuliu_edmonds.py"))
cle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cle)

rng = np.random.default_rng(2026)


def random_scores(n):
    """(n+1) x (n+1) head scores: row i = dependent, column j = head, log-softmax per row."""
    s = rng.normal(size=(n + 1, n + 1))
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


def is_single_root_tree(heads):
    """heads[i-1] = head of word i; exactly one word attached to 0 and no cycles."""
    if heads.count(0) != 1:
        return False
    n = len(heads)
    for i in range(1, n + 1):
        v, steps = i, 0
        while v != 0 and steps <= n:
            v = heads[v - 1]
            steps += 1
        if v != 0:
            return False
    return True


def exhaustive(scores):
    """Best single-root tree by enumerating all n^n head vectors."""
    n = scores.shape[0] - 1
    choices = [[j for j in range(n + 1) if j != i] for i in range(1, n + 1)]
    best, best_heads, count = -np.inf, None, 0
    for heads in itertools.product(*choices):
        if not is_single_root_tree(heads):
            continue
        count += 1
        score = sum(scores[i, heads[i - 1]] for i in range(1, n + 1))
        if score > best:
            best, best_heads = score, heads
    return best, best_heads, count


def tree_score(scores, tree):
    return sum(scores[i, tree[i]] for i in range(1, len(tree)))


def test_sentence_lengths():
    lengths, cur = [], 0
    path = os.path.join(REPO, "data", "depparse", "uz_uzudt.test.in.conllu")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur:
                    lengths.append(cur)
                cur = 0
            elif not line.startswith("#") and line.split("\t")[0].isdigit():
                cur += 1
    if cur:
        lengths.append(cur)
    return lengths


out = {"machine": f"{platform.processor() or platform.machine()} / Python {platform.python_version()} / NumPy {np.__version__}"}

# 1-2) tree count and correctness
check = {}
for n in range(2, 8):
    trials = 300 if n <= 6 else 60
    agree = 0
    for _ in range(trials):
        s = random_scores(n)
        best, _, count = exhaustive(s)
        tree = cle.chuliu_edmonds_one_root(s.copy())
        if abs(tree_score(s, tree) - best) < 1e-9 and list(tree[1:]).count(0) == 1:
            agree += 1
    check[n] = {"trials": trials, "agree": agree, "trees": count, "n^(n-1)": n ** (n - 1)}
    print("check", n, check[n], flush=True)
out["check"] = check

# 3a) exhaustive search time
exh = {}
for n in range(2, 9):
    reps = 5 if n <= 6 else 1
    s = random_scores(n)
    t0 = time.perf_counter()
    for _ in range(reps):
        exhaustive(s)
    exh[n] = (time.perf_counter() - t0) / reps
    print("exhaustive", n, exh[n], flush=True)
out["exhaustive_sec"] = exh
# seconds per enumerated head vector (n^n of them), from the two largest n
per_vector = float(np.mean([exh[n] / n ** n for n in (7, 8)]))
out["exhaustive_sec_per_vector"] = per_vector

# 3b) CLE time and greedy-step cycles
cle_sec, cycles = {}, {}
for n in [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 15, 20, 25, 30, 37, 40, 50, 60, 80, 100, 120, 150]:
    reps = 400 if n <= 40 else 100
    mats = [random_scores(n) for _ in range(reps)]
    c = []
    for m in mats:
        g = m.copy()
        cle.prepare_scores(g)
        c.append(len(cle.tarjan(np.argmax(g, axis=1))))
    t0 = time.perf_counter()
    for m in mats:
        cle.chuliu_edmonds_one_root(m)
    cle_sec[n] = (time.perf_counter() - t0) / reps
    cycles[n] = float(np.mean(c))
    print("cle", n, cle_sec[n], "cycles", cycles[n], flush=True)
out["cle_sec"] = cle_sec
out["greedy_cycles_mean"] = cycles
ns = np.array([n for n in cle_sec if n >= 20], dtype=float)
ts = np.array([cle_sec[int(n)] for n in ns])
out["cle_growth_exponent_n20_150"] = float(np.polyfit(np.log(ns), np.log(ts), 1)[0])

# 4) UzUDT test set lengths
lengths = test_sentence_lengths()
mats = [random_scores(n) for n in lengths]
t0 = time.perf_counter()
for m in mats:
    cle.chuliu_edmonds_one_root(m)
out["test"] = {
    "sentences": len(lengths), "words": int(sum(lengths)), "mean_len": float(np.mean(lengths)),
    "max_len": int(max(lengths)), "cle_total_sec": time.perf_counter() - t0,
}

# estimates for longer sentences from the measured speed per head vector
year = 365.25 * 86400
out["exhaustive_estimate"] = {
    n: {"trees": float(n) ** (n - 1), "sec": per_vector * float(n) ** n,
        "days": per_vector * float(n) ** n / 86400, "years": per_vector * float(n) ** n / year}
    for n in (9, 10, 11, 15, 19, 20, 37)
}

print(json.dumps({k: out[k] for k in ("exhaustive_sec_per_vector", "cle_growth_exponent_n20_150", "test", "exhaustive_estimate")}, indent=1))
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
