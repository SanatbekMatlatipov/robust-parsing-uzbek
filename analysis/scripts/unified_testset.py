"""
Clean unified test sets for all models.

UzUDT: sentences that are in BOTH the old test split (.1 experiments) and the official v2.17 test split
(.2 experiments), and that have no exact or near-duplicate (token Jaccard >= 0.8) in any train or dev file
used by any model. Gold annotation is taken from the official v2.17 test file.
UT: the official UT test file, filtered the same way against UT train/dev and all UzUDT train/dev files.
"""
import os

from common import (MERGED, OLD_SPLIT, RESULTS, UT, UZUDT, normalise, read_conllu, save_json, sent_id,
                    sent_text, words, write_conllu)

JACCARD = 0.8


def token_set(sent):
    return set(normalise(r[1]) for r in words(sent))


def near_duplicates(cands, pool):
    pool_sets = [token_set(p) for p in pool]
    flagged = {}
    for i, s in enumerate(cands):
        a = token_set(s)
        best = max((len(a & b) / len(a | b) for b in pool_sets if a | b), default=0.0)
        if best >= JACCARD:
            flagged[i] = round(best, 3)
    return flagged


def main():
    train_dev = []
    for p in [os.path.join(OLD_SPLIT, f"uz_uzudt.{s}.in.conllu") for s in ("train", "dev")] + \
             [os.path.join(UZUDT, f"uz_uzudt-ud-{s}.conllu") for s in ("train", "dev")] + \
             [os.path.join(UT, f"uz_ut-ud-{s}.conllu") for s in ("train", "dev")]:
        train_dev += read_conllu(p)
    exact_pool = set(normalise(sent_text(s)) for s in train_dev)

    old_test = set(normalise(sent_text(s)) for s in read_conllu(os.path.join(OLD_SPLIT, "uz_uzudt.test.in.conllu")))
    new_test = read_conllu(os.path.join(UZUDT, "uz_uzudt-ud-test.conllu"))
    shared = [s for s in new_test if normalise(sent_text(s)) in old_test]
    no_exact = [s for s in shared if normalise(sent_text(s)) not in exact_pool]
    flagged = near_duplicates(no_exact, train_dev)
    uzudt = [s for i, s in enumerate(no_exact) if i not in flagged]

    ut_test = read_conllu(os.path.join(UT, "uz_ut-ud-test.conllu"))
    ut_no_exact = [s for s in ut_test if normalise(sent_text(s)) not in exact_pool]
    ut_flagged = near_duplicates(ut_no_exact, train_dev)
    ut = [s for i, s in enumerate(ut_no_exact) if i not in ut_flagged]

    out_dir = os.path.join(RESULTS, "unified")
    os.makedirs(out_dir, exist_ok=True)
    write_conllu(uzudt, os.path.join(out_dir, "uzudt_test.gold.conllu"))
    write_conllu(ut, os.path.join(out_dir, "ut_test.gold.conllu"))

    summary = {
        "uzudt": {"official_test": len(new_test), "shared_with_old_test": len(shared),
                  "after_exact_filter": len(no_exact), "near_duplicates_removed": len(flagged),
                  "final_sentences": len(uzudt), "final_words": sum(len(words(s)) for s in uzudt),
                  "removed_ids": [sent_id(no_exact[i]) for i in flagged]},
        "ut": {"official_test": len(ut_test), "after_exact_filter": len(ut_no_exact),
               "near_duplicates_removed": len(ut_flagged), "final_sentences": len(ut),
               "final_words": sum(len(words(s)) for s in ut),
               "removed_ids": [sent_id(ut_no_exact[i]) for i in ut_flagged]},
        "jaccard_threshold": JACCARD,
    }
    print(save_json(summary, "unified_testset.json"))
    print({k: (v["final_sentences"], v["final_words"]) for k, v in summary.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
