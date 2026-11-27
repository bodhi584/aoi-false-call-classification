"""P4 experiment on the development half (days 1-64): every setting x 5 seeds -> results/p4_dev_runs.json.

Factors (P3 slide 10): S1 retraining interval, S2 training window, S3 threshold-set size m,
threshold rule (Neyman-Pearson vs. plain 1 % quantile), S5 audit share. S4 confidence is fixed at 95 %.
Forest F1-F4: imbalanced-learn 0.14 defaults. Start T0: the first day with at least 2 x 299 labelled
true defects before it (299 for the threshold set, at least as many for training).
Usage: python scripts/p4_run_dev.py [data/dataset.csv]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load, periods, simulate  # noqa: E402

GRID = {"interval": [None, 7, 14], "window": [None, 28], "m": [299, 500],
        "rule": ["np", "plain"], "audit": [1.0, 0.1]}
SEEDS = [0, 1, 2, 3, 4]
OUT = Path("results/p4_dev_runs.json")


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    days = np.unique(cal_day[:n_dev])
    start = int(next(d for d in days if y[:n_dev][cal_day[:n_dev] < d].sum() >= 2 * 299))
    res = {"start_day": start, "dev_rows": int(n_dev), "dev_true_defects": int(y[:n_dev].sum()), "runs": []}
    done = set()
    if OUT.exists():                                   # resume after an interruption
        res = json.loads(OUT.read_text())
        done = {json.dumps(r["setting"], sort_keys=True) + str(r["seed"]) for r in res["runs"]}
    for vals in itertools.product(*GRID.values()):
        s = dict(zip(GRID, vals))
        if s["interval"] is None and s["window"] is not None:
            continue                                   # a window only matters when the model is retrained
        for seed in SEEDS:
            key = json.dumps(s, sort_keys=True) + str(seed)
            if key in done:
                continue
            rec = simulate(X, y, cal_day, n_dev, start_day=start, seed=seed, **s)
            res["runs"].append({"setting": s, "seed": seed, "windows": rec, "periods": periods(rec)})
            OUT.write_text(json.dumps(res))
            print(s, seed, [round(p["missed_pct"], 1) for p in res["runs"][-1]["periods"]],
                  [round(p["released_pct"], 1) for p in res["runs"][-1]["periods"]], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
