"""Per-model LAS with 95% percentile bootstrap intervals over sentences (predicted tags, clean text, CLE)."""
import os

import numpy as np

from common import RESULTS, read_conllu, save_json, words

MODELS = ["E1.1", "E2.1", "E3.1", "E5.1", "E1.2", "E2.2", "E3.2"]
B = 10000
RNG = np.random.default_rng(2026)


def strip(rel):
    return rel.split(":")[0]


def per_sentence(model, test):
    gold = read_conllu(os.path.join(RESULTS, "unified", f"{test}_test.gold.conllu"))
    system = read_conllu(os.path.join(RESULTS, "inference", "clean", f"{model}.{test}.pred.cle.conllu"))
    correct, total = [], []
    for g, s in zip(gold, system):
        gw, sw = words(g), words(s)
        assert len(gw) == len(sw)
        correct.append(sum(a[6] == b[6] and strip(a[7]) == strip(b[7]) for a, b in zip(gw, sw)))
        total.append(len(gw))
    return np.array(correct), np.array(total)


def main():
    out = {}
    for m in MODELS:
        out[m] = {}
        for t in ("uzudt", "ut"):
            c, n = per_sentence(m, t)
            idx = RNG.integers(0, len(c), size=(B, len(c)))
            boot = 100 * c[idx].sum(axis=1) / n[idx].sum(axis=1)
            out[m][t] = {"LAS": 100 * c.sum() / n.sum(), "ci_low": float(np.percentile(boot, 2.5)),
                         "ci_high": float(np.percentile(boot, 97.5)), "sentences": int(len(c)), "words": int(n.sum())}
        print(m, {t: {k: round(v, 2) for k, v in r.items()} for t, r in out[m].items()})
    save_json(out, "las_ci.json")


if __name__ == "__main__":
    main()
