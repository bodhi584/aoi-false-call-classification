"""P4 experiment 3: the day cap on the threshold set, and one threshold per inspection type.

Grid: cap (none / 30 / 10 true defects per day) x m (299 / 500) x retraining (once / every 7 days),
Neyman-Pearson rule, all history, audit 100 %; plus one threshold per inspection type (m = 299, cap 30).
5 seeds each. Output: results/p4_dev_cap.json
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load, load_groups, periods, simulate  # noqa: E402

OUT = Path("results/p4_dev_cap.json")


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    groups = load_groups(path)
    start = json.loads(Path("results/p4_dev_runs.json").read_text())["start_day"]
    res = {"start_day": start, "runs": []}
    settings = [{"interval": i, "window": None, "m": m, "rule": "np", "audit": 1.0, "cap": c}
                for c, m, i in itertools.product([None, 30, 10], [299, 500], [None, 7])]
    settings += [{"interval": i, "window": None, "m": 299, "rule": "np", "audit": 1.0, "cap": 30, "per_type": True}
                 for i in (None, 7)]
    for s in settings:
        for seed in range(5):
            kw = {k: v for k, v in s.items() if k != "per_type"}
            rec = simulate(X, y, cal_day, n_dev, start_day=start, seed=seed,
                           groups=groups if s.get("per_type") else None, **kw)
            res["runs"].append({"setting": s, "seed": seed, "windows": rec, "periods": periods(rec)})
            OUT.write_text(json.dumps(res))
            print(s, seed, [round(p["missed_pct"], 1) for p in res["runs"][-1]["periods"]],
                  [round(p["released_pct"], 1) for p in res["runs"][-1]["periods"]], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
