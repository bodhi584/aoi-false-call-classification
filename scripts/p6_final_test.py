"""Second evaluation on the test half (data days 64-101): the improved system chosen on the development half.

This run comes AFTER the P5 final test of the frozen P4 model (results/p5_final_test.json). The improvements
were designed on the development half only, but after the P5 result was known, so this is a second look at
the test half and is reported as such. One configuration, frozen before the run, run once:
  frozen P4 system and forest (14-day schedule, all history, m = 500, NP 95 %, audit 10 %; BRF 300/sqrt/leaf 5/1:1)
  + SPC switch: AOI calls so far today > 99 % quantile of the previous 14 production days -> send to operator
  + threshold-set cap: at most 30 true defects from one day
  + burst trigger: labelled true defects today >= 3 x mean of the last 14 production days (and >= 20)
    -> extra retraining next morning
Output: results/p6_final_test.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_sim import load, periods  # noqa: E402
from p6_sim import simulate_daily  # noqa: E402
from p6_spc import spc_alarms  # noqa: E402
from p5_final_test import REF7, metrics, metrics_from_counts  # noqa: E402

SYSTEM = {"interval": 14, "m": 500, "audit": 0.1}
FOREST = {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "sampling_strategy": "all"}
IMPROVE = {"cap": 30, "burst": 3}
SPC = {"mode": "day", "q": 0.99}
OUT = Path("results/p6_final_test.json")


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    n = len(y)
    alarm = spc_alarms(path, **SPC)
    start = int(cal_day[n_dev])
    m_test = n - n_dev
    slices = [(n_dev + int(i * 0.2 * m_test), n_dev + int((i + 1) * 0.2 * m_test)) for i in range(5)]
    res = {"system": SYSTEM, "forest": FOREST, "improve": IMPROVE, "spc": SPC, "start_day": start,
           "first_row": n_dev, "slices": slices, "ref7_brfc2": REF7, "runs": []}
    for seed in range(5):
        rel = np.zeros(n, bool)
        rec = simulate_daily(X, y, cal_day, n_dev, start_day=start, seed=seed, forest=FOREST, alarm=alarm,
                             first_row=n_dev, n_end=n, out_rel=rel, **SYSTEM, **IMPROVE)
        run = {"seed": seed, "days": rec, "periods": [metrics_from_counts(p) for p in periods(rec)],
               "slices": [metrics(y[a:b], rel[a:b]) for a, b in slices], "total": metrics(y[n_dev:], rel[n_dev:])}
        res["runs"].append(run)
        OUT.write_text(json.dumps(res, indent=1))
        t = run["total"]
        print(seed, "periods missed", [round(p["missed_pct"], 2) for p in run["periods"]],
              "slices missed", [round(s["missed_pct"], 2) for s in run["slices"]],
              "| total missed %.2f released %.2f recall %.4f precision %.4f F1 %.4f" %
              (t["missed_pct"], t["released_pct"], t["recall"], t["precision"], t["f1"]), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
