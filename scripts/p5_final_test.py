"""P5 final test: the frozen model, run once on the test half (second half of the rows, data days 64-101).

Frozen in P4 on the development half only (results/p4_dev_forest.json, p4_dev_confirm.json, p4_dev_system2.json):
  system  S1 retrain every 14 days, S2 all labelled history, S3 m = 500, S4 95 % (NP rank k = 2), S5 audit 10 %
  forest  F1 300 trees, F2 sqrt features, F3 min_samples_leaf 5, F4 1:1 sampling (balanced random forest)
Every development row is labelled history; the first test row is the first row ever checked.
The first deployment day T0 is the calendar day of the first test row.

Metrics (positive = true defect, predicted positive = sent to the operator, i.e. not released):
  recall, precision, F1, missed % (= 1 - recall), released % (released false calls / false calls).
Reported per period (>= 299 true defects), per slice of [7] (5 equal chronological row slices),
and for the whole test half. 5 seeds, same as development.
Output: results/p5_final_test.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load, periods, simulate  # noqa: E402

SYSTEM = {"interval": 14, "window": None, "m": 500, "rule": "np", "audit": 0.1}
FOREST = {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "sampling_strategy": "all"}
SEEDS = range(5)
OUT = Path("results/p5_final_test.json")
# [7], Sect. VI, Table IV, model BRFC2: missed % and released % per evaluation slice (mean of 10 runs)
REF7 = {"missed": [7.3, 5.3, 8.6, 9.4, 9.3], "released": [67.5, 48.7, 65.3, 78.0, 41.5]}


def metrics(y, rel):
    """Confusion counts and scores for rows y (1 = true defect) with release decisions rel."""
    sent = ~rel
    tp = int((sent & (y == 1)).sum()); fn = int((rel & (y == 1)).sum())
    fp = int((sent & (y == 0)).sum()); tn = int((rel & (y == 0)).sum())
    recall = tp / max(tp + fn, 1); precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"td": tp + fn, "fc": fp + tn, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": recall, "precision": precision, "f1": f1,
            "missed_pct": 100 * fn / max(tp + fn, 1), "released_pct": 100 * tn / max(fp + tn, 1)}


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    n = len(y)
    start = int(cal_day[n_dev])
    m_test = n - n_dev
    slices = [(n_dev + int(i * 0.2 * m_test), n_dev + int((i + 1) * 0.2 * m_test)) for i in range(5)]
    res = {"system": SYSTEM, "forest": FOREST, "start_day": start, "first_row": n_dev, "n_rows": n,
           "slices": slices, "ref7_brfc2": REF7, "runs": []}
    for seed in SEEDS:
        rel = np.zeros(n, bool)
        rec = simulate(X, y, cal_day, n_dev, start_day=start, seed=seed, n_end=n, first_row=n_dev,
                       forest=FOREST, out_rel=rel, **SYSTEM)
        run = {"seed": seed, "windows": rec,
               "periods": [metrics_from_counts(p) for p in periods(rec)],
               "slices": [metrics(y[a:b], rel[a:b]) for a, b in slices],
               "total": metrics(y[n_dev:], rel[n_dev:])}
        res["runs"].append(run)
        OUT.write_text(json.dumps(res, indent=1))
        t = run["total"]
        print(seed, "periods missed", [round(p["missed_pct"], 2) for p in run["periods"]],
              "released", [round(p["released_pct"], 1) for p in run["periods"]],
              "| total recall %.4f precision %.4f F1 %.4f released %.2f %%" %
              (t["recall"], t["precision"], t["f1"], t["released_pct"]), flush=True)


def metrics_from_counts(p):
    """Same scores from pooled period counts (td, fc, missed, released)."""
    tp = p["td"] - p["missed"]; fp = p["fc"] - p["released"]
    recall = tp / max(p["td"], 1); precision = tp / max(tp + fp, 1)
    return p | {"recall": recall, "precision": precision,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12)}


if __name__ == "__main__":
    main(*sys.argv[1:])
