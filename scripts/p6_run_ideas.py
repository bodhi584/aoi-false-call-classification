"""P6 experiment 1: improvement ideas on the development half only (days 1-64), day-by-day simulation.

Frozen P4 system and forest; each idea is switched on alone, then all together. 5 seeds.
Periods are pooled from daily records (>= 299 true defects), a finer cut than the 14-day windows of P4.
Output: results/p6_dev_ideas.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load, periods  # noqa: E402
from p6_sim import simulate_daily  # noqa: E402

SYSTEM = {"interval": 14, "m": 500, "audit": 0.1}
FOREST = {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "sampling_strategy": "all"}
IDEAS = {
    "frozen": {},
    "burst x3": {"burst": 3},
    "burst x2": {"burst": 2},
    "burst x3 + pause 1 day": {"burst": 3, "pause": 1},
    "audit trigger": {"audit_trigger": True},
    "burst x3 + audit trigger": {"burst": 3, "audit_trigger": True},
    "threshold-set cap 30/day": {"cap": 30},
    "unseen codes never released": {"unseen": True},
    "all: burst x3 + audit + cap 30 + unseen": {"burst": 3, "audit_trigger": True, "cap": 30, "unseen": True},
}
OUT = Path("results/p6_dev_ideas.json")


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    start = json.loads(Path("results/p4_dev_runs.json").read_text())["start_day"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {"start_day": start, "system": SYSTEM,
                                                            "forest": FOREST, "runs": []}
    done = {(r["idea"], r["seed"]) for r in res["runs"]}
    for name, kw in IDEAS.items():
        for seed in range(5):
            if (name, seed) in done:
                continue
            rec = simulate_daily(X, y, cal_day, n_dev, start_day=start, seed=seed, forest=FOREST, **SYSTEM, **kw)
            P = periods(rec)
            res["runs"].append({"idea": name, "kw": kw, "seed": seed, "days": rec, "periods": P})
            OUT.write_text(json.dumps(res))
            rel = 100 * sum(r["released"] for r in rec) / sum(r["fc"] for r in rec)
            print(name, seed, [round(p["missed_pct"], 2) for p in P], "released %.2f" % rel,
                  "retrains", sum(1 for r in rec if r["retrain"]), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
