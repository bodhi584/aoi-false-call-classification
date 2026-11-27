"""P6 experiment 2: same-day safety switch and combinations (development half only, 5 seeds).
Output: results/p6_dev_ideas2.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_sim import load, periods  # noqa: E402
from p6_sim import simulate_daily  # noqa: E402
from p6_run_ideas import FOREST, SYSTEM  # noqa: E402

IDEAS = {
    "same-day switch 10": {"intraday": 10},
    "same-day switch 20": {"intraday": 20},
    "same-day switch 30": {"intraday": 30},
    "same-day switch 20 + burst x3": {"intraday": 20, "burst": 3},
    "cap 30 + burst x3": {"cap": 30, "burst": 3},
    "cap 30 + same-day switch 20": {"cap": 30, "intraday": 20},
    "cap 30 + switch 20 + burst x3": {"cap": 30, "intraday": 20, "burst": 3},
    "same-day switch 30 + burst x3": {"intraday": 30, "burst": 3},
    "cap 30 + switch 30 + burst x3": {"cap": 30, "intraday": 30, "burst": 3},
    "cap 30 + switch 40 + burst x3": {"cap": 30, "intraday": 40, "burst": 3},
}
OUT = Path("results/p6_dev_ideas2.json")


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
            print(name, seed, [round(p["missed_pct"], 2) for p in P], "released %.2f" % rel, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
