"""
Exact combinatorial baselines for uniformly random single-root dependency trees.

For y uniform on T(x) with n words:
  (A) the length d of a uniformly chosen non-root arc has P(d) = 2(n-d) / (n(n-1)), d = 1..n-1,
      E d = (n+1)/3, Var d = (n+1)(n-2)/18;
  (B) the number of dependents of a uniformly chosen word has
      P(k) = (1 - 1/n) b(k) + (1/n) b(k-1),  b = Binomial(n-2, 1/n);
  (C) the number of projective trees is Q_n = C(3n-2, n-1) / n (Lagrange inversion of T = z / (1 - T)^2);
  (D) the total arc length D has E D = (n^2-1)/3 and Var D = (n-2)(n-1)(n+1)(3n-4) / (60 n);
  (E) the number of crossing pairs of arcs, root arc included, has E C = (n-1)^2 (n-2) / (6 n),
      of which (n-1)(n-2)/(3n) involve the root arc.
  (F) Cov(D, C) = (n-2)(n-1)(n+1)(4n-7) / (180 n), slope of the best linear predictor (4n-7) / (3(3n-4));
  (G) projective trees with k head-final non-root arcs: (1/n) C(n+k-1, k) C(2n-k-2, n-1-k) (OEIS A371395),
      with root word r: t(r-1) t(n-r), t(m) = C(3m, m)/(2m+1) (OEIS A001764); all arcs head-final: (n-1)!
      trees, of which the Catalan number C(2n-2, n-1)/n are projective.
(D)-(F) follow from Moon's forest formula: a forest with components of sizes c_1..c_k (k = n - #edges) lies in a
uniform labelled tree with probability c_1 ... c_k / n^#edges.

Second null model (random linear arrangement, RLA, of a fixed tree t, the words permuted uniformly):
  P(projective) = prod_v (k_v + 1)! / n!   (k_v = number of dependents of v),
  E C = ( C(n-1, 2) - sum_v C(deg_v, 2) + n - 1 - deg_r ) / 3   (deg = degree without the root arc, r = root word),
  Var D = (n+1) ( 6 (n-1)(n-2) + (n-4) sum_v deg_v (deg_v - 1) ) / 180   (Ferrer-i-Cancho 2019, rewritten).

All laws are checked against exhaustive enumeration (U: n <= 8, Pruefer codes x choice of the root word;
RLA: all n! arrangements of random trees with n <= 7), then evaluated on the 1,184 gold trees.
"""
import heapq
import itertools
import math
import os
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

from common import RESULTS, save_json
from decoders import is_projective
from graph_stats import corpus, random_tree, tree_stats

N_BRUTE = 8          # exhaustive check up to this length
N_DIRECT = 7         # direct projectivity test (ancestor walk) up to this length; planarity test above


def arc_length_pmf(n):
    return {d: Fraction(2 * (n - d), n * (n - 1)) for d in range(1, n)}


def binom_pmf(m, p, k):
    if k < 0 or k > m:
        return Fraction(0)
    return math.comb(m, k) * p ** k * (1 - p) ** (m - k)


def outdeg_pmf(n):
    if n == 1:
        return {0: Fraction(1)}
    p = Fraction(1, n)
    return {k: (1 - p) * binom_pmf(n - 2, p, k) + p * binom_pmf(n - 2, p, k - 1) for k in range(0, n)}


def projective_count(n):
    c = math.comb(3 * n - 2, n - 1)
    assert c % n == 0
    return c // n


def total_length_moments(n):
    """Mean and variance of the total length of the non-root arcs under the uniform model."""
    return Fraction((n - 1) * (n + 1), 3), Fraction((n - 2) * (n - 1) * (n + 1) * (3 * n - 4), 60 * n)


def crossings_expectation(n):
    """Expected numbers of crossing pairs among the non-root arcs and with the root arc (uniform model)."""
    return Fraction((n - 1) * (n - 2) * (n - 3), 6 * n), Fraction((n - 1) * (n - 2), 3 * n)


def total_length_crossings_cov(n):
    return Fraction((n - 2) * (n - 1) * (n + 1) * (4 * n - 7), 180 * n)


def projective_by_direction(n, k):
    """Projective trees on n words with exactly k head-final non-root arcs."""
    if n == 1:
        return 1 if k == 0 else 0
    c = math.comb(n + k - 1, k) * math.comb(2 * n - k - 2, n - 1 - k)
    assert c % n == 0
    return c // n


def ternary(m):
    return math.comb(3 * m, m) // (2 * m + 1)


def projective_by_root(n, r):
    """Projective trees on n words whose root word is r."""
    return ternary(r - 1) * ternary(n - r)


def head_final_arcs(heads):
    return sum(1 for d, h in enumerate(heads, start=1) if h != 0 and h > d)


def crossings(heads):
    """Number of crossing pairs of arcs, root arc included; heads[i-1] = head of word i, 0 = root."""
    spans = [(min(h, d), max(h, d)) for d, h in enumerate(heads, start=1)]
    return sum(1 for (a, b), (c, e) in itertools.combinations(spans, 2) if a < c < b < e or c < a < e < b)


def rla_projective_prob(heads):
    n = len(heads)
    k = Counter(h for h in heads if h != 0)
    num = 1
    for v in range(1, n + 1):
        num *= math.factorial(k.get(v, 0) + 1)
    return Fraction(num, math.factorial(n))


def rla_expected_crossings(heads):
    n = len(heads)
    deg = Counter()
    root = heads.index(0) + 1
    for d, h in enumerate(heads, start=1):
        if h != 0:
            deg[d] += 1
            deg[h] += 1
    pairs = math.comb(n - 1, 2) - sum(math.comb(deg[v], 2) for v in range(1, n + 1))
    return Fraction(pairs + n - 1 - deg[root], 3)


def rla_length_variance(heads):
    n = len(heads)
    deg = Counter()
    for d, h in enumerate(heads, start=1):
        if h != 0:
            deg[d] += 1
            deg[h] += 1
    adjacent = sum(k * (k - 1) for k in deg.values())
    return Fraction((n + 1) * (6 * (n - 1) * (n - 2) + (n - 4) * adjacent), 180)


def rla_check(n, trials, rng):
    """Exhaustive check of the RLA laws on random trees: all n! arrangements of each tree."""
    for _ in range(trials):
        heads = random_tree(n)
        proj = cross = D1 = D2 = 0
        for perm in itertools.permutations(range(1, n + 1)):
            pos = (0,) + perm                          # word v is placed at position pos[v]
            new = [0] * n
            for d, h in enumerate(heads, start=1):
                new[pos[d] - 1] = pos[h]
            proj += bool(is_projective([0] + new))
            cross += crossings(new)
            D = sum(abs(h - d) for d, h in enumerate(new, start=1) if h != 0)
            D1 += D
            D2 += D * D
        f = math.factorial(n)
        if Fraction(proj, f) != rla_projective_prob(heads) or Fraction(cross, f) != rla_expected_crossings(heads):
            return False
        if Fraction(D1, f) != Fraction((n - 1) * (n + 1), 3) or Fraction(D2, f) - Fraction(D1, f) ** 2 != rla_length_variance(heads):
            return False
    return True


def pruefer_edges(seq, n):
    if n == 2:
        return [(1, 2)]
    degree = [1] * (n + 1)
    for x in seq:
        degree[x] += 1
    leaves = [i for i in range(1, n + 1) if degree[i] == 1]
    heapq.heapify(leaves)
    edges = []
    for x in seq:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, x))
        degree[x] -= 1
        if degree[x] == 1:
            heapq.heappush(leaves, x)
    edges.append((heapq.heappop(leaves), heapq.heappop(leaves)))
    return edges


def orient(edges, n, root):
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    heads = [0] * (n + 1)
    stack, seen = [root], {root}
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in seen:
                seen.add(u)
                heads[u] = v
                stack.append(u)
    heads[root] = 0
    return heads


def crossing_free(edges):
    spans = [(min(a, b), max(a, b)) for a, b in edges]
    for (a, b), (c, d) in itertools.combinations(spans, 2):
        if a < c < b < d or c < a < d < b:
            return False
    return True


def brute_force(n):
    """Exact totals over all n^(n-1) single-root trees."""
    arc_len = Counter()
    outdeg = Counter()
    projective = 0
    trees = 0
    D1 = D2 = cross_nonroot = cross_root = DC = 0
    by_dir, by_root = Counter(), Counter()
    all_hf = all_hf_proj = 0
    for seq in itertools.product(range(1, n + 1), repeat=max(n - 2, 0)):
        edges = pruefer_edges(seq, n) if n >= 2 else []
        deg = Counter()
        spans = [(min(a, b), max(a, b)) for a, b in edges]
        D = sum(b - a for a, b in spans)
        D1 += n * D
        D2 += n * D * D
        cn = sum(1 for (a, b), (c, e) in itertools.combinations(spans, 2) if a < c < b < e or c < a < e < b)
        cross_nonroot += n * cn
        for r in range(1, n + 1):
            cr = sum(1 for a, b in spans if a < r < b)
            cross_root += cr
            DC += D * (cn + cr)
            heads = orient(edges, n, r)[1:] if n >= 2 else [0]
            hf = head_final_arcs(heads)
            all_hf += hf == n - 1
            if cn + cr == 0:                      # projective (no crossings, root arc included)
                by_dir[hf] += 1
                by_root[r] += 1
                all_hf_proj += hf == n - 1
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
            arc_len[abs(a - b)] += n          # the same n-1 arcs for each of the n root choices
        for v in range(1, n + 1):
            outdeg[deg[v] - 1] += n - 1       # v is not the root word
            outdeg[deg[v]] += 1               # v is the root word
        if n <= N_DIRECT:
            for r in range(1, n + 1):
                projective += bool(is_projective(orient(edges, n, r)))
        elif crossing_free(edges):
            # planar trees: projective iff the root word lies under no arc
            projective += sum(1 for r in range(1, n + 1) if not any(min(a, b) < r < max(a, b) for a, b in edges))
        trees += n
    return trees, arc_len, outdeg, projective, (D1, D2, cross_nonroot, cross_root, DC, by_dir, by_root, all_hf, all_hf_proj)


def main():
    checks = []
    for n in range(1, N_BRUTE + 1):
        trees, arc_len, outdeg, proj, (D1, D2, cn, cr, DC, by_dir, by_root, all_hf, all_hf_proj) = brute_force(n)
        ok_count = trees == n ** (n - 1)
        ED, VD = total_length_moments(n)
        EC_n, EC_r = crossings_expectation(n)
        ok_D = n < 2 or (Fraction(D1, trees) == ED and Fraction(D2, trees) - ED * ED == VD)
        ok_C = n < 2 or (Fraction(cn, trees) == EC_n and Fraction(cr, trees) == EC_r)
        ok_cov = n < 2 or Fraction(DC, trees) - ED * (EC_n + EC_r) == total_length_crossings_cov(n)
        ok_dir = all(by_dir[k] == projective_by_direction(n, k) for k in range(n))
        ok_root = all(by_root[r] == projective_by_root(n, r) for r in range(1, n + 1))
        ok_hf = all_hf == math.factorial(n - 1) and all_hf_proj == math.comb(2 * n - 2, n - 1) // n
        ok_arc = all(Fraction(arc_len.get(d, 0), trees * (n - 1)) == p for d, p in arc_length_pmf(n).items()) if n > 1 else True
        ok_deg = all(Fraction(outdeg.get(k, 0), trees * n) == p for k, p in outdeg_pmf(n).items())
        ok_proj = proj == projective_count(n)
        checks.append({"n": n, "trees": trees, "projective": proj, "formula": projective_count(n),
                       "arc_pmf_ok": ok_arc, "outdeg_pmf_ok": ok_deg, "count_ok": ok_count, "projective_ok": ok_proj,
                       "total_length_ok": ok_D, "crossings_ok": ok_C, "covariance_ok": ok_cov,
                       "direction_ok": ok_dir, "root_position_ok": ok_root, "head_final_ok": ok_hf})
        print(checks[-1], flush=True)
    assert all(c["arc_pmf_ok"] and c["outdeg_pmf_ok"] and c["count_ok"] and c["projective_ok"]
               and c["total_length_ok"] and c["crossings_ok"] and c["covariance_ok"] and c["direction_ok"]
               and c["root_position_ok"] and c["head_final_ok"] for c in checks)
    rla_ok = {n: rla_check(n, 40, None) for n in range(2, 8)}
    print("RLA checks", rla_ok, flush=True)
    assert all(rla_ok.values())

    # variance identity Var d = (n+1)(n-2)/18
    for n in range(2, 60):
        pmf = arc_length_pmf(n)
        m1 = sum(d * p for d, p in pmf.items())
        m2 = sum(d * d * p for d, p in pmf.items())
        assert m1 == Fraction(n + 1, 3) and m2 - m1 * m1 == Fraction((n + 1) * (n - 2), 18)

    # mixtures over the sentence lengths of the gold trees
    trees_all = corpus("UzUDT") + corpus("UT")
    lengths = [len(h) for h, _ in trees_all]
    arc_mix = Counter()
    deg_mix = Counter()
    for n in lengths:
        if n > 1:
            for d, p in arc_length_pmf(n).items():
                arc_mix[d] += float(p) * (n - 1)
        for k, p in outdeg_pmf(n).items():
            deg_mix[k] += float(p) * n
    tot_arcs = sum(n - 1 for n in lengths)
    tot_words = sum(lengths)
    arc_mix = {d: v / tot_arcs for d, v in sorted(arc_mix.items())}
    deg_mix = {k: v / tot_words for k, v in sorted(deg_mix.items())}
    proj_share_random = float(np.mean([projective_count(n) / n ** (n - 1) for n in lengths]))

    # observed projective share and exact random share by length band
    bands = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 200)]
    by_band = []
    for lo, hi in bands:
        sel = [h for h, _ in trees_all if lo <= len(h) <= hi]
        if not sel:
            continue
        obs = float(np.mean([is_projective([0] + h) for h in sel]))
        rnd = float(np.mean([projective_count(len(h)) / len(h) ** (len(h) - 1) for h in sel]))
        by_band.append({"band": f"{lo}-{hi}" if hi < 200 else f">{lo - 1}", "sentences": len(sel),
                        "observed_projective": obs, "random_projective_exact": rnd})
    by_n = []
    for n in sorted(set(lengths)):
        sel = [h for h, _ in trees_all if len(h) == n]
        by_n.append({"n": n, "sentences": len(sel),
                     "observed_projective": float(np.mean([is_projective([0] + h) for h in sel])),
                     "random_projective_exact": projective_count(n) / n ** (n - 1)})

    # crossings, total length and projectivity under both null models, per sentence
    per_sent = []
    labels = ["UzUDT"] * len(corpus("UzUDT")) + ["UT"] * len(corpus("UT"))
    for (h, _), label in zip(trees_all, labels):
        n = len(h)
        if n < 2:
            continue
        ED, VD = total_length_moments(n)
        EC_n, EC_r = crossings_expectation(n)
        D = sum(abs(hh - d) for d, hh in enumerate(h, start=1) if hh != 0)
        per_sent.append({"n": n, "C": crossings(h), "EC_U": float(EC_n + EC_r), "EC_RLA": float(rla_expected_crossings(h)),
                         "P_RLA": float(rla_projective_prob(h)), "P_U": projective_count(n) / n ** (n - 1),
                         "proj": bool(is_projective([0] + h)), "D": D, "ED": float(ED),
                         "z_D": float((D - ED) / math.sqrt(VD)) if VD > 0 else None,
                         "z_D_RLA": float((D - ED) / math.sqrt(rla_length_variance(h))) if rla_length_variance(h) > 0 else None,
                         "corpus": label})
    null_by_band = []
    for lo, hi in bands:
        sel = [r for r in per_sent if lo <= r["n"] <= hi]
        if not sel:
            continue
        null_by_band.append({"band": f"{lo}-{hi}" if hi < 200 else f">{lo - 1}", "sentences": len(sel),
                             "observed_projective": float(np.mean([r["proj"] for r in sel])),
                             "rla_projective": float(np.mean([r["P_RLA"] for r in sel])),
                             "uniform_projective": float(np.mean([r["P_U"] for r in sel])),
                             "observed_crossings": float(np.mean([r["C"] for r in sel])),
                             "rla_crossings": float(np.mean([r["EC_RLA"] for r in sel])),
                             "uniform_crossings": float(np.mean([r["EC_U"] for r in sel]))})
    zs = [r["z_D"] for r in per_sent if r["z_D"] is not None]
    null_summary = {
        "sentences": len(per_sent),
        "observed_crossings_total": sum(r["C"] for r in per_sent),
        "rla_crossings_total": sum(r["EC_RLA"] for r in per_sent),
        "uniform_crossings_total": sum(r["EC_U"] for r in per_sent),
        "observed_projective_share": float(np.mean([r["proj"] for r in per_sent])),
        "rla_projective_share": float(np.mean([r["P_RLA"] for r in per_sent])),
        "uniform_projective_share": float(np.mean([r["P_U"] for r in per_sent])),
        "sentences_with_crossings": int(sum(r["C"] > 0 for r in per_sent)),
        "z_total_length_mean": float(np.mean(zs)), "z_total_length_median": float(np.median(zs)),
        "share_z_below_minus2": float(np.mean(np.array(zs) < -2)), "share_z_below_0": float(np.mean(np.array(zs) < 0)),
        "z_rla_mean": float(np.mean([r["z_D_RLA"] for r in per_sent if r["z_D_RLA"] is not None])),
        "z_rla_median": float(np.median([r["z_D_RLA"] for r in per_sent if r["z_D_RLA"] is not None])),
        "share_z_rla_below_minus2": float(np.mean([r["z_D_RLA"] < -2 for r in per_sent if r["z_D_RLA"] is not None])),
        "share_z_rla_below_0": float(np.mean([r["z_D_RLA"] < 0 for r in per_sent if r["z_D_RLA"] is not None])),
        "by_band": null_by_band,
        "by_corpus": {c: {"sentences": len(sel),
                          "observed_crossings_mean": float(np.mean([r["C"] for r in sel])),
                          "uniform_crossings_mean": float(np.mean([r["EC_U"] for r in sel])),
                          "rla_crossings_mean": float(np.mean([r["EC_RLA"] for r in sel])),
                          "observed_projective": float(np.mean([r["proj"] for r in sel])),
                          "uniform_projective": float(np.mean([r["P_U"] for r in sel])),
                          "rla_projective": float(np.mean([r["P_RLA"] for r in sel]))}
                      for c, sel in (("UzUDT", [r for r in per_sent if r["corpus"] == "UzUDT"]),
                                     ("UT", [r for r in per_sent if r["corpus"] == "UT"]),
                                     ("UzUDT+UT", per_sent))},
    }
    print("null models", {k: v for k, v in null_summary.items() if k != "by_band"})
    for r in null_by_band:
        print(r)
    by_n_null = []
    for n in sorted(set(r["n"] for r in per_sent)):
        sel = [r for r in per_sent if r["n"] == n]
        by_n_null.append((n, len(sel), np.mean([r["proj"] for r in sel]), np.mean([r["P_RLA"] for r in sel]),
                          projective_count(n) / n ** (n - 1), np.mean([r["C"] for r in sel]),
                          np.mean([r["EC_RLA"] for r in sel]), sel[0]["EC_U"]))

    # head-final arcs and root position of the projective gold trees against uniformly random projective trees
    dir_rows, root_rows, pred_rows = [], [], []
    for (h, upos), label in zip(trees_all, labels):
        n = len(h)
        if n < 2:
            continue
        K = head_final_arcs(h)
        r = h.index(0) + 1
        ED, VD = total_length_moments(n)
        D = sum(abs(hh - d) for d, hh in enumerate(h, start=1) if hh != 0)
        kappa = Fraction(4 * n - 7, 3 * (3 * n - 4))
        pred_rows.append(float(sum(crossings_expectation(n)) + kappa * (D - ED)))
        last_word = max(i for i in range(1, n + 1) if upos[i - 1] != "PUNCT") if any(u != "PUNCT" for u in upos) else n
        root_rows.append({"n": n, "root": r, "root_is_last": r == n, "root_is_last_word": r == last_word,
                          "p_last_U": 1 / n, "p_last_proj": projective_by_root(n, n) / projective_count(n),
                          "p_lastword_U": 1 / n, "p_lastword_proj": projective_by_root(n, last_word) / projective_count(n)})
        if not is_projective([0] + h):
            continue
        Q = projective_count(n)
        tail = sum(projective_by_direction(n, k) for k in range(K, n)) / Q
        mean = (n - 1) / 2
        second = sum(k * k * projective_by_direction(n, k) for k in range(n)) / Q
        dir_rows.append({"n": n, "K": K, "share": K / (n - 1), "tail": tail, "var_null": second - mean * mean})
    direction = {
        "projective_sentences": len(dir_rows),
        "head_final_share_observed": sum(r["K"] for r in dir_rows) / sum(r["n"] - 1 for r in dir_rows),
        "z_total": (sum(r["K"] for r in dir_rows) - sum((r["n"] - 1) / 2 for r in dir_rows)) / math.sqrt(sum(r["var_null"] for r in dir_rows)),
        "median_tail": float(np.median([r["tail"] for r in dir_rows if r["n"] >= 3])),
        "share_tail_below_005": float(np.mean([r["tail"] < 0.05 for r in dir_rows if r["n"] >= 3])),
        "share_all_head_final": float(np.mean([r["K"] == r["n"] - 1 for r in dir_rows])),
        "mean_null_prob_all_head_final": float(np.mean([math.comb(2 * r["n"] - 2, r["n"] - 1) / r["n"] / projective_count(r["n"]) for r in dir_rows])),
    }
    root = {
        "sentences": len(root_rows),
        "observed_root_last": float(np.mean([r["root_is_last"] for r in root_rows])),
        "observed_root_last_word": float(np.mean([r["root_is_last_word"] for r in root_rows])),
        "U_root_last": float(np.mean([r["p_last_U"] for r in root_rows])),
        "proj_root_last": float(np.mean([r["p_last_proj"] for r in root_rows])),
        "proj_root_last_word": float(np.mean([r["p_lastword_proj"] for r in root_rows])),
        "proj_root_at_ends_n11": 2 * projective_by_root(11, 11) / projective_count(11),
    }
    linear = {"predicted_crossings_mean": float(np.mean(pred_rows)), "predicted_negative_share": float(np.mean(np.array(pred_rows) < 0)),
              "uniform_crossings_mean": float(np.mean([float(sum(crossings_expectation(len(h)))) for h, _ in trees_all if len(h) >= 2])),
              "observed_crossings_mean": null_summary["by_corpus"]["UzUDT+UT"]["observed_crossings_mean"]}
    # the same on content-word trees: punctuation tokens are leaves in UD, so deleting them leaves a tree
    cw_dir, cw_root, skipped = [], [], 0
    for h, upos in trees_all:
        keep = [i for i in range(1, len(h) + 1) if upos[i - 1] != "PUNCT"]
        pos = {old: new for new, old in enumerate(keep, start=1)}
        if any(h[i - 1] != 0 and h[i - 1] not in pos for i in keep) or len(keep) < 2:
            skipped += 1
            continue
        hh = [0 if h[i - 1] == 0 else pos[h[i - 1]] for i in keep]
        n = len(hh)
        r = hh.index(0) + 1
        cw_root.append({"n": n, "last": r == n, "first": r == 1, "p_last_proj": projective_by_root(n, n) / projective_count(n)})
        if not is_projective([0] + hh):
            continue
        K = head_final_arcs(hh)
        Q = projective_count(n)
        tail = sum(projective_by_direction(n, k) for k in range(K, n)) / Q
        second = sum(k * k * projective_by_direction(n, k) for k in range(n)) / Q
        cw_dir.append({"n": n, "K": K, "tail": tail, "var_null": second - ((n - 1) / 2) ** 2,
                       "all_hf": K == n - 1, "p_all_hf": math.comb(2 * n - 2, n - 1) / n / Q})
    content_words = {
        "sentences": len(cw_root), "skipped": skipped, "projective": len(cw_dir),
        "head_final_share": sum(r["K"] for r in cw_dir) / sum(r["n"] - 1 for r in cw_dir),
        "z_total": (sum(r["K"] for r in cw_dir) - sum((r["n"] - 1) / 2 for r in cw_dir)) / math.sqrt(sum(r["var_null"] for r in cw_dir)),
        "median_tail": float(np.median([r["tail"] for r in cw_dir if r["n"] >= 3])),
        "share_tail_below_005": float(np.mean([r["tail"] < 0.05 for r in cw_dir if r["n"] >= 3])),
        "share_all_head_final": float(np.mean([r["all_hf"] for r in cw_dir])),
        "null_all_head_final": float(np.mean([r["p_all_hf"] for r in cw_dir])),
        "root_last": float(np.mean([r["last"] for r in cw_root])), "root_first": float(np.mean([r["first"] for r in cw_root])),
        "root_last_U": float(np.mean([1 / r["n"] for r in cw_root])),
        "root_last_proj": float(np.mean([r["p_last_proj"] for r in cw_root])),
        "mean_n": float(np.mean([r["n"] for r in cw_root])),
    }
    direction["content_words"] = content_words
    # figure data: share of head-final arcs among projective content-word trees, observed and exact null mixture
    edges_bins = np.linspace(0, 1, 11)
    obs_hist = np.zeros(10)
    null_hist = np.zeros(10)
    for r in cw_dir:
        if r["n"] < 3:
            continue
        b = min(int(r["K"] / (r["n"] - 1) * 10), 9)
        obs_hist[b] += 1
        Q = projective_count(r["n"])
        for k in range(r["n"]):
            null_hist[min(int(k / (r["n"] - 1) * 10), 9)] += projective_by_direction(r["n"], k) / Q
    obs_hist /= obs_hist.sum()
    null_hist /= null_hist.sum()
    with open(os.path.join(RESULTS, "csv", "head_final_share.csv"), "w") as f:
        f.write("bin_low,bin_high,observed,null_projective\n")
        for i in range(10):
            f.write(f"{edges_bins[i]:.1f},{edges_bins[i + 1]:.1f},{obs_hist[i]:.6f},{null_hist[i]:.6f}\n")
    with open(os.path.join(RESULTS, "csv", "root_position.csv"), "w") as f:
        f.write("model,first,last\n")
        f.write(f"U,{content_words['root_last_U']:.6f},{content_words['root_last_U']:.6f}\n")
        f.write(f"projective,{content_words['root_last_proj']:.6f},{content_words['root_last_proj']:.6f}\n")
        f.write(f"Uzbek,{content_words['root_first']:.6f},{content_words['root_last']:.6f}\n")
    content_words["limit_root_last_proj"] = [projective_by_root(m, m) / projective_count(m) for m in (11, 20, 40, 80, 160)]
    print("content words", content_words)
    print("direction", direction)
    print("root", root)
    print("linear prediction of crossings", linear)

    # observed out-degree distribution (all words) for the comparison table
    obs_deg = Counter()
    for h, upos in trees_all:
        _, _, _, _, degrees = tree_stats(h, upos)
        obs_deg.update(degrees)
    obs_deg = {k: v / tot_words for k, v in sorted(obs_deg.items())}

    out = {
        "brute_force_checks": checks,
        "projective_counts": {n: projective_count(n) for n in range(1, 21)},
        "projective_fraction": {n: projective_count(n) / n ** (n - 1) for n in range(1, 41)},
        "growth_rate_27_over_4": [projective_count(n + 1) / projective_count(n) for n in (10, 20, 40, 80)],
        "corpus_mixture": {
            "arc_length_pmf": arc_mix,
            "outdegree_pmf": deg_mix,
            "expected_random_projective_share": proj_share_random,
            "expected_random_mean_arc_length": sum(d * p for d, p in arc_mix.items()),
            "observed_outdegree_pmf": obs_deg,
        },
        "projective_by_band": by_band,
        "projective_by_n": by_n,
        "null_models": null_summary,
        "rla_checks": {str(k): v for k, v in rla_ok.items()},
        "total_length_moments": {n: [float(x) for x in total_length_moments(n)] for n in range(2, 41)},
        "direction": direction, "root_position": root, "linear_prediction": linear,
        "covariance_D_C": {n: float(total_length_crossings_cov(n)) for n in range(2, 41)},
        "expected_crossings": {n: float(sum(crossings_expectation(n))) for n in range(2, 41)},
    }
    save_json(out, "exact_baselines.json")
    # CSV for the figures
    with open(os.path.join(RESULTS, "csv", "exact_arc_length.csv"), "w") as f:
        f.write("d,random_exact\n")
        for d, p in arc_mix.items():
            f.write(f"{d},{p:.6g}\n")
    with open(os.path.join(RESULTS, "csv", "exact_outdegree.csv"), "w") as f:
        f.write("k,observed,random_exact\n")
        for k in range(0, 12):
            f.write(f"{k},{obs_deg.get(k, 0):.6g},{deg_mix.get(k, 0):.6g}\n")
    with open(os.path.join(RESULTS, "csv", "projective_by_n.csv"), "w") as f:
        f.write("n,sentences,observed,random_exact\n")
        for r in by_n:
            f.write(f"{r['n']},{r['sentences']},{r['observed_projective']:.6g},{r['random_projective_exact']:.6g}\n")
    with open(os.path.join(RESULTS, "csv", "null_models_by_n.csv"), "w") as f:
        f.write("n,sentences,observed_projective,rla_projective,uniform_projective,observed_crossings,rla_crossings,uniform_crossings\n")
        for row in by_n_null:
            f.write(",".join([str(row[0]), str(row[1])] + [f"{x:.6g}" for x in row[2:]]) + "\n")
    print("expected random projective share", proj_share_random)
    print("expected random mean arc length", out["corpus_mixture"]["expected_random_mean_arc_length"])
    print(by_band)


if __name__ == "__main__":
    main()
