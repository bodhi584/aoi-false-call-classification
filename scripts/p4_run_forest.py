"""P4 experiment 4: forest parameters F1-F4 (P3 slide 10: "whether to tune them: experimental design").

System parameters fixed to the winner of experiment 1 among the settings with retraining
(S1 every 14 days, S2 all history, S3 m = 500, S4 95 %, S5 audit 10 %, Neyman-Pearson rule).
Grid: F1 trees 100 / 300, F2 features per split sqrt (14) / log2 (8) / 0.25 (51),
F3 minimum leaf 1 / 5, F4 sampling 1:1 / 1:2. 5 seeds each. Output: results/p4_dev_forest.json
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load, periods, simulate  # noqa: E402

SYSTEM = {"interval": 14, "window": None, "m": 500, "rule": "np", "audit": 0.1}
GRID = {"n_estimators": [100, 300], "max_features": ["sqrt", "log2", 0.25],
        "min_samples_leaf": [1, 5], "sampling_strategy": ["all", 0.5]}
OUT = Path("results/p4_dev_forest.json")


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    start = json.loads(Path("results/p4_dev_runs.json").read_text())["start_day"]
    res = {"start_day": start, "system": SYSTEM, "runs": []}
    done = set()
    if OUT.exists():
        res = json.loads(OUT.read_text())
        done = {json.dumps(r["setting"]["forest"], sort_keys=True) + str(r["seed"]) for r in res["runs"]}
    for vals in itertools.product(*GRID.values()):
        f = dict(zip(GRID, vals))
        for seed in range(5):
            if json.dumps(f, sort_keys=True) + str(seed) in done:
                continue
            rec = simulate(X, y, cal_day, n_dev, start_day=start, seed=seed, forest=f, **SYSTEM)
            res["runs"].append({"setting": SYSTEM | {"forest": f}, "seed": seed, "windows": rec,
                                "periods": periods(rec)})
            OUT.write_text(json.dumps(res))
            print(f, seed, [round(p["missed_pct"], 2) for p in res["runs"][-1]["periods"]],
                  [round(p["released_pct"], 1) for p in res["runs"][-1]["periods"]], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
