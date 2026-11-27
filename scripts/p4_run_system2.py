"""P4 experiment 6: threshold-set size m (S3) and audit share (S5) with the tuned forest.

Forest = winner of experiment 4. Grid: S1 every 7 / 14 days, S2 all history,
S3 m = 500 / 750 / 1000, S5 audit 5 / 10 / 20 %, NP rule at 95 %. 5 seeds.
Start day stays at T0 = 40 so all settings are scored on the same check windows;
with a large m the first steps may have too few training true defects and release nothing.
Output: results/p4_dev_system2.json
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load, periods, simulate  # noqa: E402

FOREST = {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "sampling_strategy": "all"}
OUT = Path("results/p4_dev_system2.json")


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    start = json.loads(Path("results/p4_dev_runs.json").read_text())["start_day"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {"start_day": start, "forest": FOREST, "runs": []}
    done = {(json.dumps(r["setting"], sort_keys=True), r["seed"]) for r in res["runs"]}
    for interval, m, audit in itertools.product([14, 7], [500, 750, 1000], [0.05, 0.1, 0.2]):
        s = {"interval": interval, "window": None, "m": m, "rule": "np", "audit": audit, "forest": FOREST}
        for seed in range(5):
            if (json.dumps(s, sort_keys=True), seed) in done:
                continue
            sim = {k: v for k, v in s.items() if k != "forest"}
            rec = simulate(X, y, cal_day, n_dev, start_day=start, seed=seed, forest=FOREST, **sim)
            res["runs"].append({"setting": s, "seed": seed, "windows": rec, "periods": periods(rec)})
            OUT.write_text(json.dumps(res))
            print(interval, m, audit, seed, [round(p["missed_pct"], 2) for p in res["runs"][-1]["periods"]],
                  [round(p["released_pct"], 1) for p in res["runs"][-1]["periods"]], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
