"""P4 experiment 8 (28 Nov, development half only): threshold rules and retraining schedules on the FROZEN system
(tuned BRF, m = 500, audit 10 %), so that every P3 choice is compared against its alternatives in the same setting.
Rules: classical 0.5 cut, [7] cross-validated threshold, plain 1 % quantile (NP 95 % / 99 % are in experiment 7).
Retraining: train once, every 28 d (every 7 / 14 d and the 28-day window are in experiment 5). 5 seeds.
Output: results/p4_dev_proof.json. Does not change the final-test result.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load, periods, simulate  # noqa: E402

BRF = {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "sampling_strategy": "all"}
BASE = {"interval": 14, "window": None, "m": 500, "rule": "np", "audit": 0.1}
SETTINGS = {
    "threshold: 0.5 cut": BASE | {"rule": "half"},
    "threshold: [7] cross-validated": BASE | {"rule": "cv"},
    "threshold: plain 1 % quantile": BASE | {"rule": "plain"},
    "retraining: train once": BASE | {"interval": None},
    "retraining: every 28 d": BASE | {"interval": 28},
}
OUT = Path("results/p4_dev_proof.json")


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    start = json.loads(Path("results/p4_dev_runs.json").read_text())["start_day"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {"start_day": start, "forest": BRF, "runs": []}
    done = {(r["name"], r["seed"]) for r in res["runs"]}
    for name, s in SETTINGS.items():
        for seed in range(5):
            if (name, seed) in done:
                continue
            rec = simulate(X, y, cal_day, n_dev, start_day=start, seed=seed, forest=BRF, **s)
            P = periods(rec)
            res["runs"].append({"name": name, "setting": s, "seed": seed, "windows": rec, "periods": P})
            OUT.write_text(json.dumps(res))
            rel = 100 * sum(w["released"] for w in rec) / sum(w["fc"] for w in rec)
            print(name, seed, [round(p["missed_pct"], 2) for p in P], "released %.2f" % rel, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
