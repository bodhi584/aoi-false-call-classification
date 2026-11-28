"""P4 experiment 9 (28 Nov, development half only): one-parameter sweeps around the frozen model,
to draw tuning curves (released share and worst missed vs. one parameter, the others fixed).

Frozen: every 14 d, all history, m 500, NP 95 %, audit 10 %; BRF 300 trees, max_features sqrt, min leaf 5, 1:1.
Only points not already in experiments 4-8 are run here; make_results_figures.py merges them.
5 seeds. Output: results/p4_dev_sweep.json. Does not change the frozen model or the final-test result.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load, periods, simulate  # noqa: E402

SYS = {"interval": 14, "window": None, "m": 500, "rule": "np", "audit": 0.1}
BRF = {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "sampling_strategy": "all"}
POINTS = ([("min_samples_leaf", v) for v in (2, 3, 8, 12, 20, 40)]
          + [("n_estimators", v) for v in (50, 200, 500)]
          + [("max_features", v) for v in (0.1, 0.5)]
          + [("m", v) for v in (300, 400, 600)]
          + [("interval", v) for v in (10, 21)]
          + [("audit", v) for v in (0.15, 0.3)])
OUT = Path("results/p4_dev_sweep.json")


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    start = json.loads(Path("results/p4_dev_runs.json").read_text())["start_day"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {"start_day": start, "runs": []}
    done = {(r["param"], r["value"], r["seed"]) for r in res["runs"]}
    for p, v in POINTS:
        s, f = dict(SYS), dict(BRF)
        (f if p in BRF else s)[p] = v
        for seed in range(5):
            if (p, v, seed) in done:
                continue
            rec = simulate(X, y, cal_day, n_dev, start_day=start, seed=seed, forest=f, **s)
            P = periods(rec)
            res["runs"].append({"param": p, "value": v, "setting": s, "forest": f, "seed": seed,
                                "windows": rec, "periods": P})
            OUT.write_text(json.dumps(res))
            rel = 100 * sum(w["released"] for w in rec) / sum(w["fc"] for w in rec)
            print(p, v, seed, [round(q["missed_pct"], 2) for q in P], "released %.2f" % rel, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
