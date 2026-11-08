"""
Graph-theoretic statistics of the 1,184 gold Uzbek dependency trees (UzUDT v2.17 + UT), compared with
uniformly random single-root trees on the same vertex sets.

Random trees: uniform on T(x) <=> a uniform labelled tree on {1..n} (random Pruefer code) plus a uniform
root word, oriented away from the root. Exact law: E[mean arc length] = (n+1)/3 for every n >= 2.
"""
import heapq
import math
import os
from collections import Counter, defaultdict

import numpy as np
from scipy import optimize

from common import RESULTS, UT, UZUDT, read_conllu, save_json, words

RNG = np.random.default_rng(2026)
MC_PER_SENTENCE = 200


def trees(path):
    """(heads, upos) per sentence; heads[i-1] = head of word i (0 = root)."""
    out = []
    for s in read_conllu(path):
        w = words(s)
        out.append(([int(r[6]) for r in w], [r[3] for r in w]))
    return out


def corpus(name):
    base = UZUDT if name == "UzUDT" else UT
    prefix = "uz_uzudt" if name == "UzUDT" else "uz_ut"
    res = []
    for split in ("train", "dev", "test"):
        res += trees(os.path.join(base, f"{prefix}-ud-{split}.conllu"))
    return res


def children_lists(heads):
    ch = defaultdict(list)
    for d, h in enumerate(heads, start=1):
        ch[h].append(d)
    return ch


def depths(heads):
    ch = children_lists(heads)
    depth = {0: 0}
    stack = [0]
    while stack:
        v = stack.pop()
        for c in ch[v]:
            depth[c] = depth[v] + 1
            stack.append(c)
    return depth


def is_projective(heads):
    n = len(heads)
    # ancestors via parent pointers
    for d, h in enumerate(heads, start=1):
        if h == 0:
            continue
        lo, hi = min(h, d), max(h, d)
        for k in range(lo + 1, hi):
            v = k
            while v != 0 and v != h:
                v = heads[v - 1]
            if v != h:
                return False
    return True


def random_tree(n):
    """Uniform single-root arborescence on words 1..n (root arc 0 -> r)."""
    if n == 1:
        return [0]
    if n == 2:
        edges = [(1, 2)]
    else:
        seq = list(RNG.integers(1, n + 1, size=n - 2))
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
        u, v = heapq.heappop(leaves), heapq.heappop(leaves)
        edges.append((u, v))
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    root = int(RNG.integers(1, n + 1))
    heads = [None] * n
    heads[root - 1] = 0
    stack, seen = [root], {root}
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in seen:
                seen.add(u)
                heads[u - 1] = v
                stack.append(u)
    return heads


def tree_stats(heads, upos=None, drop_punct=False):
    arcs = []
    for d, h in enumerate(heads, start=1):
        if h == 0:
            continue
        if drop_punct and upos is not None and upos[d - 1] == "PUNCT":
            continue
        arcs.append((h, d))
    lengths = [abs(h - d) for h, d in arcs]
    head_right = sum(1 for h, d in arcs if h > d)
    depth = depths(heads)
    outdeg = Counter(h for h in heads if h != 0)
    degrees = [outdeg.get(i, 0) for i in range(1, len(heads) + 1)]
    return lengths, head_right, len(arcs), max(depth.values()), degrees


def summarise(tree_list, label):
    n_sent = len(tree_list)
    n_words = sum(len(h) for h, _ in tree_list)
    lens_all, lens_np = [], []
    right_np = arcs_np = 0
    nonproj = 0
    expected_sum = observed_sum = 0.0
    for heads, upos in tree_list:
        la, _, _, _, _ = tree_stats(heads)
        ln, hr, na, _, _ = tree_stats(heads, upos, drop_punct=True)
        lens_all += la
        lens_np += ln
        right_np += hr
        arcs_np += na
        nonproj += 0 if is_projective(heads) else 1
        n = len(heads)
        if n >= 2:
            observed_sum += sum(la)
            expected_sum += (n - 1) * (n + 1) / 3
    return {
        "corpus": label, "sentences": n_sent, "words": n_words, "mean_length": n_words / n_sent,
        "arcs_no_root_no_punct": arcs_np,
        "mean_arc_length_no_punct": float(np.mean(lens_np)),
        "adjacent_share_no_punct": float(np.mean(np.array(lens_np) == 1)),
        "head_right_share_no_punct": right_np / arcs_np,
        "nonprojective_trees": nonproj, "nonprojective_share": nonproj / n_sent,
        "mean_arc_length_all_nonroot": float(np.mean(lens_all)),
        "arc_law_expected_total": expected_sum, "observed_total": observed_sum,
        "observed_over_random": observed_sum / expected_sum,
    }


KERNELS = {
    "geometric": (lambda d, q: (d - 1) * np.log1p(-q), (1e-4, 0.999)),
    "cauchy": (lambda d, s: -np.log1p(((d - 1) / s) ** 2), (0.05, 50.0)),
    "power": (lambda d, g: -g * np.log(d), (0.0, 10.0)),
}


def fit_arc_length(arcs):
    """Fits of the arc-length law conditioned on the sentence length: an arc of a sentence of n words has
    P(d | n) = (n - d) k(d) / sum_{e=1}^{n-1} (n - e) k(e), d = 1..n-1, where n - d is the number of word pairs at
    distance d (k = 1 gives the uniform model U). Kernels: geometric (1-q)^(d-1),
    Cauchy-type (1 + (d-1)^2/s^2)^(-1) and power d^(-g); composite likelihood over all arcs."""
    arcs = np.asarray(arcs)
    d, n = arcs[:, 0], arcs[:, 1]
    lengths = np.unique(n)
    per_n = {m: np.bincount(d[n == m], minlength=m)[1:m] for m in lengths}   # counts of d = 1..m-1

    def nll(logk):
        total = 0.0
        for m, cnt in per_n.items():
            e = np.arange(1, m)
            lw = np.log(m - e) + logk(e)
            lz = np.logaddexp.reduce(lw)
            total -= float((cnt * (lw - lz)).sum())
        return total

    out = {"n_arcs": int(len(d)), "null": {"neg_loglik": nll(lambda e: np.zeros(len(e)))}}
    out["null"]["aic"] = 2 * out["null"]["neg_loglik"]
    for name, (logk, bounds) in KERNELS.items():
        r = optimize.minimize_scalar(lambda t: nll(lambda e: logk(e, t)), bounds=bounds, method="bounded",
                                     options={"xatol": 1e-8})
        out[name] = {"param": float(r.x), "neg_loglik": float(r.fun), "aic": 2 + 2 * float(r.fun)}
    # pooled distributions implied by the fitted laws (arc-weighted mixture over sentence lengths)
    D = int(n.max()) - 1
    pooled = {}
    for name in ["null"] + list(KERNELS):
        mix = np.zeros(D + 1)
        logk = (lambda e: np.zeros(len(e))) if name == "null" else (lambda e, f=KERNELS[name][0], t=out[name]["param"]: f(e, t))
        for m, cnt in per_n.items():
            e = np.arange(1, m)
            lw = np.log(m - e) + logk(e)
            mix[1:m] += cnt.sum() * np.exp(lw - np.logaddexp.reduce(lw))
        pooled[name] = mix[1:] / len(d)
    observed = np.bincount(d, minlength=D + 1)[1:] / len(d)
    return out, np.arange(1, D + 1), observed, pooled


def main():
    uz, ut = corpus("UzUDT"), corpus("UT")
    both = uz + ut
    table = [summarise(uz, "UzUDT"), summarise(ut, "UT"), summarise(both, "UzUDT+UT")]

    # Monte Carlo random trees with the observed sentence lengths
    mc_len, mc_right, mc_arcs, mc_nonproj, mc_trees = [], 0, 0, 0, 0
    mc_height = defaultdict(list)
    mc_deg = Counter()
    for heads, _ in both:
        n = len(heads)
        for _ in range(MC_PER_SENTENCE):
            r = random_tree(n)
            la, hr, na, h, deg = tree_stats(r)
            mc_len += la
            mc_right += hr
            mc_arcs += na
            mc_nonproj += 0 if is_projective(r) else 1
            mc_trees += 1
            mc_height[n].append(h)
            mc_deg.update(deg)
    random_row = {"corpus": "random trees (Monte Carlo)", "trees": mc_trees,
                  "mean_arc_length_all_nonroot": float(np.mean(mc_len)),
                  "head_right_share": mc_right / mc_arcs, "nonprojective_share": mc_nonproj / mc_trees}

    # check of the arc-length law per length: Monte Carlo mean vs (n+1)/3
    arc_law = []
    for n in (5, 10, 20, 40):
        sample = []
        for _ in range(4000):
            la, _, _, _, _ = tree_stats(random_tree(n))
            sample.append(np.mean(la))
        arc_law.append({"n": n, "monte_carlo": float(np.mean(sample)), "exact": (n + 1) / 3,
                       "std_error": float(np.std(sample) / math.sqrt(len(sample)))})

    # observed curves by length
    by_n = defaultdict(list)
    height_obs = defaultdict(list)
    deg_obs = Counter()
    lens_np, arcs_np = [], []
    for heads, upos in both:
        la, _, _, h, deg = tree_stats(heads)
        ln, _, _, _, _ = tree_stats(heads, upos, drop_punct=True)
        lens_np += ln
        arcs_np += [(x, len(heads)) for x in ln]
        if len(heads) >= 2:
            by_n[len(heads)].append(np.mean(la))
        height_obs[len(heads)].append(h)
        deg_obs.update(deg)

    os.makedirs(os.path.join(RESULTS, "csv"), exist_ok=True)
    with open(os.path.join(RESULTS, "csv", "mean_arc_by_n.csv"), "w") as f:
        f.write("n,observed,ci_low,ci_high,random,sentences\n")
        for n in sorted(by_n):
            v = np.array(by_n[n])
            if len(v) < 5 or n > 40:
                continue
            se = v.std(ddof=1) / math.sqrt(len(v))
            f.write(f"{n},{v.mean():.4f},{v.mean() - 1.96 * se:.4f},{v.mean() + 1.96 * se:.4f},{(n + 1) / 3:.4f},{len(v)}\n")
    with open(os.path.join(RESULTS, "csv", "height_by_n.csv"), "w") as f:
        f.write("n,observed,random,sentences\n")
        for n in sorted(height_obs):
            if len(height_obs[n]) < 5 or n > 40:
                continue
            f.write(f"{n},{np.mean(height_obs[n]):.4f},{np.mean(mc_height[n]):.4f},{len(height_obs[n])}\n")
    tot_obs, tot_mc = sum(deg_obs.values()), sum(mc_deg.values())
    with open(os.path.join(RESULTS, "csv", "outdegree.csv"), "w") as f:
        f.write("k,observed,random,poisson1\n")
        for k in range(0, 9):
            f.write(f"{k},{deg_obs.get(k, 0) / tot_obs:.5f},{mc_deg.get(k, 0) / tot_mc:.5f},{math.exp(-1) / math.factorial(k):.5f}\n")

    fits, support, emp, pooled = fit_arc_length(arcs_np)
    with open(os.path.join(RESULTS, "csv", "arc_length_hist.csv"), "w") as f:
        f.write("d,observed,geometric,cauchy,power,random\n")
        for i, d in enumerate(support[:15]):
            f.write(f"{d},{emp[i]:.6f},{pooled['geometric'][i]:.6f},{pooled['cauchy'][i]:.6f},"
                    f"{pooled['power'][i]:.6f},{pooled['null'][i]:.6f}\n")

    tail = {"observed_share_gt10": float(np.mean(np.array(lens_np) > 10))}
    obs_deg_mean = sum(k * v for k, v in deg_obs.items()) / tot_obs
    summary = {"corpora": table, "random_trees": random_row, "arc_law_check": arc_law,
               "arc_length_fits_no_punct": fits, "tail": tail,
               "outdegree": {"observed_max_share_k": deg_obs.most_common(3), "observed_mean": obs_deg_mean,
                             "observed_share_k0": deg_obs.get(0, 0) / tot_obs, "random_share_k0": mc_deg.get(0, 0) / tot_mc},
               "height": {"observed_mean": float(np.mean([h for v in height_obs.values() for h in v])),
                          "random_mean": float(np.mean([h for v in mc_height.values() for h in v]))}}
    print(save_json(summary, "graph_stats.json"))
    for row in table:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    print(random_row)
    print(arc_law)
    print(fits)


if __name__ == "__main__":
    main()
