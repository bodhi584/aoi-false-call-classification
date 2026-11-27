"""P4 experiment 2: threshold rules side by side, and the day cap on the threshold set.

Same rolling setting for all (weekly retraining, all history, audit 100 %), 5 seeds:
  half  classical rule: defect if s(x) >= 0.5
  cv    threshold method of [7]: 5-fold CV, highest t with recall >= 99 %, mean of the folds
  plain 1 % quantile of the latest m = 299 true-defect scores
  np    Neyman-Pearson rank [14, Alg. 1] on the latest m = 299 true defects
  np + cap 30 / 50: same, but at most 30 / 50 of the 299 from one calendar day
Output: results/p4_dev_rules.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load, periods, simulate  # noqa: E402

VARIANTS = [("half", None), ("cv", None), ("plain", None), ("np", None), ("np", 50), ("np", 30)]
OUT = Path("results/p4_dev_rules.json")


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    start = json.loads(Path("results/p4_dev_runs.json").read_text())["start_day"]   # same T0 as experiment 1
    res = {"start_day": start, "runs": []}
    for rule, cap in VARIANTS:
        for seed in range(5):
            s = {"interval": 7, "window": None, "m": 299, "rule": rule, "audit": 1.0, "cap": cap}
            rec = simulate(X, y, cal_day, n_dev, start_day=start, seed=seed, **s)
            res["runs"].append({"setting": s, "seed": seed, "windows": rec, "periods": periods(rec)})
            OUT.write_text(json.dumps(res))
            print(rule, cap, seed, [round(p["missed_pct"], 1) for p in res["runs"][-1]["periods"]],
                  [round(p["released_pct"], 1) for p in res["runs"][-1]["periods"]], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
