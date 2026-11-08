"""Training dynamics of the parsers from their CSV logs (dev set, every 100 Adam steps)."""
import csv
import os

from common import REPO, RESULTS, save_json

LOGS = {"E1.1": "uz_uzudt_E1.1", "E2.1": "uz_uzudt_E2.1", "E3.1": "uz_uzudt_E3.1", "E5.1": "uz_uzudt_E5.1",
        "E1.2": "uz_combined_E1.2", "E2.2": "uz_combined_E2.2"}
PATIENCE = 3000


def main():
    summary, long_rows = {}, []
    for name, stem in LOGS.items():
        path = os.path.join(REPO, "saved_models", "depparse", f"{stem}_parser_training_log.csv")
        rows = list(csv.DictReader(open(path)))
        best = max(rows, key=lambda r: float(r["las"]))
        first = rows[0]
        summary[name] = {"steps": int(rows[-1]["step"]), "total_time_s": float(rows[-1]["elapsed_sec"]),
                         "best_step": int(best["step"]), "best_dev_LAS": float(best["las"]),
                         "time_to_best_s": float(best["elapsed_sec"]),
                         "loss_at_100": float(first["train_loss"]), "loss_at_best": float(best["train_loss"]),
                         "loss_min": min(float(r["train_loss"]) for r in rows),
                         "LAS_at_100": float(first["las"]), "LAS_last": float(rows[-1]["las"]),
                         "amsgrad_switch_step": int(best["step"]) + PATIENCE,
                         "stop_step": int(best["step"]) + 2 * PATIENCE,
                         "sec_per_step": float(rows[-1]["elapsed_sec"]) / int(rows[-1]["step"])}
        for r in rows:
            long_rows.append((name, int(r["step"]), float(r["train_loss"]), float(r["las"])))
    os.makedirs(os.path.join(RESULTS, "csv"), exist_ok=True)
    for name in LOGS:
        with open(os.path.join(RESULTS, "csv", f"curve_{name}.csv"), "w") as f:
            f.write("step,loss,las\n")
            for n, s, l, a in long_rows:
                if n == name:
                    f.write(f"{s},{l:.4f},{a:.2f}\n")
    save_json(summary, "training_curves.json")
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
