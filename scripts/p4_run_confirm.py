"""P4 experiment 5: re-check the system parameters with the tuned forest (two-stage search, stage 2 -> 1).

Forest = winner of experiment 4 (300 trees, sqrt features, minimum leaf 5, sampling 1:1).
System grid: S1 every 7 / 14 days x S2 all history / 28 days; S3 m = 500, S5 audit 10 %, NP rule. 5 seeds.
Output: results/p4_dev_confirm.json
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load, periods, simulate  # noqa: E402

FOREST = {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "sampling_strategy": "all"}
OUT = Path("results/p4_dev_confirm.json")


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    start = json.loads(Path("results/p4_dev_runs.json").read_text())["start_day"]
    res = {"start_day": start, "forest": FOREST, "runs": []}
    for interval, window in itertools.product([7, 14], [None, 28]):
        s = {"interval": interval, "window": window, "m": 500, "rule": "np", "audit": 0.1}
        for seed in range(5):
            rec = simulate(X, y, cal_day, n_dev, start_day=start, seed=seed, forest=FOREST, **s)
            res["runs"].append({"setting": s | {"forest": FOREST}, "seed": seed, "windows": rec, "periods": periods(rec)})
            OUT.write_text(json.dumps(res))
            print(s, seed, [round(p["missed_pct"], 2) for p in res["runs"][-1]["periods"]],
                  [round(p["released_pct"], 1) for p in res["runs"][-1]["periods"]], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
