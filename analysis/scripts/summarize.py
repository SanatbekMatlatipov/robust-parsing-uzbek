"""Summaries of the inference results: accuracy, decoders and the non-projective subset."""
import json
import os

import numpy as np

from common import RESULTS, read_conllu, save_json, words
from decoders import is_projective

MODELS = ["E1.1", "E2.1", "E3.1", "E5.1", "E1.2", "E2.2", "E3.2"]
LABEL = {"E1.1": "FastText", "E2.1": "TahrirchiBERT, last", "E3.1": "TahrirchiBERT, mean", "E5.1": "BERTbek, mean",
         "E1.2": "FastText", "E2.2": "TahrirchiBERT, last", "E3.2": "TahrirchiBERT, mean"}
DECODERS = ["greedy", "cle_multi_root", "cle_one_root", "eisner", "mbr"]
TESTS = {"uzudt": "uzudt_test.gold.conllu", "ut": "ut_test.gold.conllu"}


def load(m):
    with open(os.path.join(RESULTS, f"inference_{m}.json")) as f:
        return json.load(f)


def main():
    rec = {m: load(m) for m in MODELS}
    acc_rows = []
    for m in MODELS:
        r = rec[m]["results"]
        row = {"model": m, "vectors": LABEL[m], "data": "UzUDT" if m.endswith(".1") else "UzUDT+UT",
               "dev_LAS_reproduced": rec[m]["dev_check"]["LAS"], "dev_LAS_logged": rec[m]["dev_logged_best_LAS"]}
        for t in TESTS:
            for tags in ("gold", "pred"):
                o = r[f"{t}.clean.{tags}_tags"]["official_cle_one_root"]
                row[f"{t}_{tags}_UAS"], row[f"{t}_{tags}_LAS"], row[f"{t}_{tags}_CLAS"] = o["UAS"], o["LAS"], o["CLAS"]
            row[f"{t}_UPOS"] = round(r[f"{t}.clean.pred_tags"]["tagging"]["UPOS"], 2)
            row[f"{t}_UFeats"] = round(r[f"{t}.clean.pred_tags"]["tagging"]["UFeats"], 2)
            if f"{t}.matched.pred_tags" in r:
                row[f"{t}_matched_pred_LAS"] = r[f"{t}.matched.pred_tags"]["official_cle_one_root"]["LAS"]
                row[f"{t}_matched_gold_LAS"] = r[f"{t}.matched.gold_tags"]["official_cle_one_root"]["LAS"]
        acc_rows.append(row)

    dec = {}
    for t in TESTS:
        for tags in ("pred", "gold"):
            key = f"{t}.clean.{tags}_tags"
            dec[key] = {}
            for d in DECODERS:
                vals = [rec[m]["results"][key]["decoders"][d] for m in MODELS]
                dec[key][d] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
                dec[key][d]["valid_min"] = float(min(v["valid_tree_share"] for v in vals))
                dec[key][d]["LAS_min"] = float(min(v["LAS"] for v in vals))
                dec[key][d]["LAS_max"] = float(max(v["LAS"] for v in vals))

    # LAS on gold non-projective sentences: exact non-projective decoding vs projective DP
    nonproj = {}
    for t, f in TESTS.items():
        gold = read_conllu(os.path.join(RESULTS, "unified", f))
        gheads = [np.array([0] + [int(r[6]) for r in words(s)]) for s in gold]
        grels = [[r[7].split(":")[0] for r in words(s)] for s in gold]
        idx = [i for i, h in enumerate(gheads) if not is_projective(h)]
        nonproj[t] = {"nonprojective_sentences": len(idx), "sentences": len(gold), "per_model": {}}
        for m in MODELS:
            z = np.load(os.path.join(RESULTS, "inference", "clean", f"{m}.{t}.pred.npz"), allow_pickle=True)
            res = {}
            for d in ("cle_one_root", "eisner"):
                tot = las = uas = 0
                for i in idx:
                    h, rel = z[f"heads_{d}"][i], z[f"rels_{d}"][i]
                    for k in range(1, len(gheads[i])):
                        tot += 1
                        if h[k] == gheads[i][k]:
                            uas += 1
                            las += int(str(rel[k - 1]).split(":")[0] == grels[i][k - 1])
                res[d] = {"UAS": 100 * uas / tot, "LAS": 100 * las / tot, "words": tot}
            nonproj[t]["per_model"][m] = res
        for d in ("cle_one_root", "eisner"):
            nonproj[t][f"mean_LAS_{d}"] = float(np.mean([nonproj[t]["per_model"][m][d]["LAS"] for m in MODELS]))
            nonproj[t][f"mean_UAS_{d}"] = float(np.mean([nonproj[t]["per_model"][m][d]["UAS"] for m in MODELS]))

    save_json({"accuracy": acc_rows, "decoders_mean_over_models": dec, "nonprojective_subset": nonproj}, "summary.json")
    os.makedirs(os.path.join(RESULTS, "csv"), exist_ok=True)
    with open(os.path.join(RESULTS, "csv", "accuracy.csv"), "w") as f:
        keys = list(acc_rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in acc_rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    for r in acc_rows:
        print(r["model"], r["vectors"], r["data"], "| UzUDT pred LAS", r["uzudt_pred_LAS"], "UT pred LAS", r["ut_pred_LAS"],
              "| gold:", r["uzudt_gold_LAS"], r["ut_gold_LAS"], "| UPOS", r["uzudt_UPOS"], r["ut_UPOS"])
    for key, v in dec.items():
        print(key, {d: (round(x["LAS"], 2), round(x["valid_tree_share"], 1), round(x["ms_per_sentence"], 3)) for d, x in v.items()})
    for t, v in nonproj.items():
        print(t, v["nonprojective_sentences"], "nonproj sentences; LAS CLE", round(v["mean_LAS_cle_one_root"], 2),
              "Eisner", round(v["mean_LAS_eisner"], 2), "UAS", round(v["mean_UAS_cle_one_root"], 2), round(v["mean_UAS_eisner"], 2))


if __name__ == "__main__":
    main()
