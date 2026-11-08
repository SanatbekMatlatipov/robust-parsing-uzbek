"""
Arc-length laws fitted by composite likelihood on the content-word trees of the 1184 gold sentences (punctuation
tokens, which are leaves in UD, removed and the positions renumbered, so that the number of word pairs at the
distance d is n_c - d with n_c the number of content words), compared by the composite likelihood information
criterion CLIC = -2 l_C + 2 tr(H^{-1} J) (Varin and Vidoni, 2005; Varin et al., 2011), where the sensitivity H and
the variability J are estimated from the per-sentence score contributions (sentences are independent, arcs within
a sentence are not). The ordinary AIC and the fits on the original positions are kept for reference.
Output: results/kernel_clic.json
"""
import numpy as np
from scipy import optimize

from common import save_json
from graph_stats import corpus

KERNELS = {
    "geometric": (lambda d, q: (d - 1) * np.log1p(-q), (1e-4, 0.999)),
    "cauchy": (lambda d, s: -np.log1p(((d - 1) / s) ** 2), (0.05, 50.0)),
    "power": (lambda d, g: -g * np.log(d), (0.0, 10.0)),
}


def content_word_trees(trees):
    """(heads, n) on the content words; punctuation removed and positions renumbered."""
    out, skipped = [], 0
    for heads, upos in trees:
        keep = [i for i in range(1, len(heads) + 1) if upos[i - 1] != "PUNCT"]
        pos = {old: new for new, old in enumerate(keep, start=1)}
        if len(keep) < 2 or any(heads[i - 1] != 0 and heads[i - 1] not in pos for i in keep):
            skipped += 1
            continue
        out.append([0 if heads[i - 1] == 0 else pos[heads[i - 1]] for i in keep])
    return out, skipped


def sentence_arcs(trees_cw):
    """Per sentence: (array of arc lengths of the non-root arcs, n)."""
    return [(np.array([abs(h - d) for d, h in enumerate(hh, start=1) if h != 0]), len(hh)) for hh in trees_cw]


def original_position_arcs(trees):
    out = []
    for heads, upos in trees:
        n = len(heads)
        lengths = [abs(h - d) for d, h in enumerate(heads, start=1) if h != 0 and upos[d - 1] != "PUNCT"]
        if lengths:
            out.append((np.array(lengths), n))
    return out


def sentence_loglik(arcs, logk):
    """Composite log-likelihood contributions per sentence for P(d | n) = (n - d) k(d) / sum_e (n - e) k(e)."""
    vals = []
    for d, n in arcs:
        e = np.arange(1, n)
        lw = np.log(n - e) + logk(e)
        lz = np.logaddexp.reduce(lw)
        vals.append(float((np.log(n - d) + logk(d) - lz).sum()))
    return np.array(vals)


def fit(arcs):
    out = {"arcs": int(sum(len(d) for d, _ in arcs)), "sentences": len(arcs)}
    null = sentence_loglik(arcs, lambda e: np.zeros(len(e)))
    out["uniform"] = {"loglik": float(null.sum()), "aic": float(-2 * null.sum()), "clic": float(-2 * null.sum())}
    for name, (logk, bounds) in KERNELS.items():
        f = lambda t: -sentence_loglik(arcs, lambda e: logk(e, t)).sum()
        r = optimize.minimize_scalar(f, bounds=bounds, method="bounded", options={"xatol": 1e-9})
        t = float(r.x)
        h = 1e-4 * max(1.0, abs(t))
        l0 = sentence_loglik(arcs, lambda e: logk(e, t))
        lp = sentence_loglik(arcs, lambda e: logk(e, t + h))
        lm = sentence_loglik(arcs, lambda e: logk(e, t - h))
        g = (lp - lm) / (2 * h)                       # per-sentence scores
        h2 = (lp - 2 * l0 + lm) / (h * h)             # per-sentence second derivatives
        J, H = float((g * g).sum()), float(-h2.sum())
        out[name] = {"param": t, "loglik": float(l0.sum()), "aic": float(-2 * l0.sum() + 2),
                     "penalty_clic": 2 * J / H, "clic": float(-2 * l0.sum() + 2 * J / H),
                     "sensitivity_H": H, "variability_J": J, "score_sum": float(g.sum())}
    return out


def describe(arcs):
    d = np.concatenate([x for x, _ in arcs])
    return {"arcs": int(len(d)), "mean_length": float(d.mean()), "adjacent_share": float(np.mean(d == 1)),
            "share_gt10": float(np.mean(d > 10))}


def main():
    uz, ut = corpus("UzUDT"), corpus("UT")
    out = {}
    for label, trees in (("UzUDT", uz), ("UT", ut), ("UzUDT+UT", uz + ut)):
        cw, skipped = content_word_trees(trees)
        arcs = sentence_arcs(cw)
        out[label] = {"content_word_trees": len(cw), "skipped": skipped, "describe": describe(arcs)}
        if label == "UzUDT+UT":
            out[label]["fits_content_words"] = fit(arcs)
            orig = original_position_arcs(trees)
            out[label]["describe_original_positions"] = describe(orig)
            out[label]["fits_original_positions"] = fit(orig)
    print({k: v["describe"] for k, v in out.items()})
    for key in ("fits_content_words", "fits_original_positions"):
        print(key)
        for name, r in out["UzUDT+UT"][key].items():
            if isinstance(r, dict):
                print("  ", name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    save_json(out, "kernel_clic.json")


if __name__ == "__main__":
    main()
