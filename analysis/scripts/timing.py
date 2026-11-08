"""
Decoding time against sentence length:
  * real score matrices of the 7 models (unified test sets, predicted tags) for every decoder,
  * random log-softmax matrices (hard case for CLE: many greedy cycles) for CLE and Eisner,
  * exhaustive enumeration (measured n <= 8, extrapolated) from results/exhaustive_timing.json,
    written by exhaustive_timing.py.
"""
import glob
import json
import os
import time
from collections import defaultdict

import numpy as np

from common import RESULTS, save_json
from decoders import cle_one_root, eisner, greedy, mbr

RNG = np.random.default_rng(2026)


def random_scores(n):
    s = RNG.normal(size=(n + 1, n + 1))
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


def fit_exponent(ns, ts):
    ns, ts = np.asarray(ns, float), np.asarray(ts, float)
    return float(np.polyfit(np.log(ns), np.log(ts), 1)[0])


def main():
    real = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(RESULTS, "inference", "clean", "*.pred.npz")):
        z = np.load(f, allow_pickle=True)
        for d in ("greedy", "cle_one_root", "eisner", "mbr"):
            for n, t in zip(z["n"], z[f"time_{d}"]):
                real[d][int(n)].append(float(t))
    rows = []
    for n in sorted(real["cle_one_root"]):
        if n > 40 or len(real["cle_one_root"][n]) < 10:
            continue
        rows.append([n] + [1000 * np.mean(real[d][n]) for d in ("greedy", "cle_one_root", "eisner", "mbr")]
                    + [len(real["cle_one_root"][n])])
    os.makedirs(os.path.join(RESULTS, "csv"), exist_ok=True)
    with open(os.path.join(RESULTS, "csv", "time_real.csv"), "w") as f:
        f.write("n,greedy_ms,cle_ms,eisner_ms,mbr_ms,count\n")
        for r in rows:
            f.write(",".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in r) + "\n")

    rnd = {"n": [], "cle_ms": [], "eisner_ms": []}
    for n in (5, 8, 11, 15, 20, 25, 30, 37, 40, 50, 60):
        mats = [random_scores(n) for _ in range(60)]
        t0 = time.perf_counter()
        for m in mats:
            cle_one_root(m)
        tc = (time.perf_counter() - t0) / len(mats)
        t0 = time.perf_counter()
        for m in mats[:20]:
            eisner(m)
        te = (time.perf_counter() - t0) / 20
        rnd["n"].append(n)
        rnd["cle_ms"].append(1000 * tc)
        rnd["eisner_ms"].append(1000 * te)
    with open(os.path.join(RESULTS, "csv", "time_random.csv"), "w") as f:
        f.write("n,cle_ms,eisner_ms\n")
        for n, c, e in zip(rnd["n"], rnd["cle_ms"], rnd["eisner_ms"]):
            f.write(f"{n},{c:.5f},{e:.5f}\n")

    measured = json.load(open(os.path.join(RESULTS, "exhaustive_timing.json")))
    per_vec = measured["exhaustive_sec_per_vector"]
    with open(os.path.join(RESULTS, "csv", "time_exhaustive.csv"), "w") as f:
        f.write("n,measured_s,estimated_s\n")
        for n in range(2, 21):
            meas = measured["exhaustive_sec"].get(str(n))
            f.write(f"{n},{meas if meas is not None else 'nan'},{per_vec * float(n) ** n:.6e}\n")

    big = [r for r in rows if r[0] >= 10]
    summary = {
        "real_mean_ms": {d: 1000 * float(np.mean([t for v in real[d].values() for t in v]))
                         for d in ("greedy", "cle_one_root", "eisner", "mbr")},
        "real_exponent_n10_40": {d: fit_exponent([r[0] for r in big], [r[i] for r in big])
                                 for i, d in ((2, "cle_one_root"), (3, "eisner"), (4, "mbr"))},
        "random_exponent": {"cle_one_root": fit_exponent(rnd["n"][4:], rnd["cle_ms"][4:]),
                            "eisner": fit_exponent(rnd["n"][4:], rnd["eisner_ms"][4:])},
        "random_ms": rnd,
        "exhaustive_sec_per_vector": per_vec,
        "exhaustive_measured_s": measured["exhaustive_sec"],
        "cle_random_ms": {k: 1000 * v for k, v in measured["cle_sec"].items()},
        "machine": measured["machine"],
    }
    save_json(summary, "timing.json")
    print(json.dumps({k: summary[k] for k in ("real_mean_ms", "real_exponent_n10_40", "random_exponent")}, indent=1))


if __name__ == "__main__":
    main()
