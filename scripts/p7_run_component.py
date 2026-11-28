"""P7 (28 Nov): component-code monitoring, development half only (days 1-64), day-by-day simulation, 5 seeds.

Follows the root-cause finding (results/p4_root_cause.json): true defects come in bursts on a few component codes.
Frozen system and BRF; added per component code (meta_feat1):
  - component switch: same-day p-chart of operator-confirmed true defects per component -> stop releasing it
  - component cap: at most N threshold-set true defects from one component
  - component SPC: label-free, AOI calls of a component so far today > 99 % quantile of its previous 14 days
The test half is NOT used (it was already evaluated twice). Output: results/p7_dev_component.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load, periods  # noqa: E402
from p6_sim import simulate_daily  # noqa: E402
from p6_spc import spc_alarms  # noqa: E402

SYSTEM = {"interval": 14, "m": 500, "audit": 0.1}
FOREST = {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "sampling_strategy": "all"}
IDEAS = {
    "frozen": {},
    "component switch 3σ, ≥ 5": {"comp_switch": 3, "comp_min": 5},
    "component switch 3σ, ≥ 3": {"comp_switch": 3, "comp_min": 3},
    "component switch 5σ, ≥ 5": {"comp_switch": 5, "comp_min": 5},
    "component cap 100": {"comp_cap": 100},
    "component cap 50": {"comp_cap": 50},
    "component SPC on calls (label-free)": {"alarm": "comp_spc"},
    "component switch 3σ + component cap 100": {"comp_switch": 3, "comp_min": 5, "comp_cap": 100},
    "component switch 3σ + burst retraining": {"comp_switch": 3, "comp_min": 5, "burst": 3},
    "component switch 3σ + component cap 100 + burst retraining": {"comp_switch": 3, "comp_min": 5, "comp_cap": 100, "burst": 3},
    "component SPC q95 (label-free)": {"alarm": "comp_spc95"},
    "component SPC + burst retraining": {"alarm": "comp_spc", "burst": 3},
    "component SPC + day cap 30 + burst retraining": {"alarm": "comp_spc", "cap": 30, "burst": 3},
    "component SPC + day cap 30": {"alarm": "comp_spc", "cap": 30},
}
OUT = Path("results/p7_dev_component.json")


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    comp = pd.read_csv(path, usecols=["meta_feat1"])["meta_feat1"].astype("category").cat.codes.to_numpy()
    alarms = {"comp_spc": spc_alarms(path, by_code=True, mode="day", q=0.99),
              "comp_spc95": spc_alarms(path, by_code=True, mode="day", q=0.95)}
    start = json.loads(Path("results/p4_dev_runs.json").read_text())["start_day"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {"start_day": start, "system": SYSTEM, "forest": FOREST, "runs": []}
    done = {(r["idea"], r["seed"]) for r in res["runs"]}
    for name, kw in IDEAS.items():
        for seed in range(5):
            if (name, seed) in done:
                continue
            k = dict(kw)
            if "alarm" in k:
                k["alarm"] = alarms[k["alarm"]]
            rec = simulate_daily(X, y, cal_day, n_dev, start_day=start, seed=seed, forest=FOREST, comp=comp, **SYSTEM, **k)
            P = periods(rec)
            res["runs"].append({"idea": name, "kw": kw, "seed": seed, "days": rec, "periods": P})
            OUT.write_text(json.dumps(res))
            rel = 100 * sum(r["released"] for r in rec) / sum(r["fc"] for r in rec)
            print(name, seed, [round(p["missed_pct"], 2) for p in P], "released %.2f" % rel, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
