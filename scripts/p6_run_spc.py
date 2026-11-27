"""P6 experiment 3: SPC switch on the AOI call stream (label-free), development half only, 5 seeds.
Output: results/p6_dev_spc.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_sim import load, periods  # noqa: E402
from p6_sim import simulate_daily  # noqa: E402
from p6_spc import spc_alarms  # noqa: E402
from p6_run_ideas import FOREST, SYSTEM  # noqa: E402

ALARMS = {"spc day-total q95": dict(mode="day", q=0.95), "spc day-total q99": dict(mode="day", q=0.99),
          "spc 60-min q95": dict(mode="rolling", q=0.95)}
IDEAS = {
    "spc day-total q95": ("spc day-total q95", {}),
    "spc day-total q99": ("spc day-total q99", {}),
    "spc 60-min q95": ("spc 60-min q95", {}),
    "spc day-total q99 + cap 30 + burst x3": ("spc day-total q99", {"cap": 30, "burst": 3}),
    "spc day-total q95 + cap 30 + burst x3": ("spc day-total q95", {"cap": 30, "burst": 3}),
    "spc day-total q99 + cap 30 + switch 30 + burst x3": ("spc day-total q99", {"cap": 30, "intraday": 30, "burst": 3}),
    "spc day-total q99 + cap 30 + switch 50 + burst x3": ("spc day-total q99", {"cap": 30, "intraday": 50, "burst": 3}),
}
OUT = Path("results/p6_dev_spc.json")


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    alarms = {k: spc_alarms(path, **v) for k, v in ALARMS.items()}
    start = json.loads(Path("results/p4_dev_runs.json").read_text())["start_day"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {"start_day": start, "system": SYSTEM,
                                                            "forest": FOREST, "alarms": ALARMS, "runs": []}
    done = {(r["idea"], r["seed"]) for r in res["runs"]}
    for name, (al, kw) in IDEAS.items():
        for seed in range(5):
            if (name, seed) in done:
                continue
            rec = simulate_daily(X, y, cal_day, n_dev, start_day=start, seed=seed, forest=FOREST,
                                 alarm=alarms[al], **SYSTEM, **kw)
            P = periods(rec)
            res["runs"].append({"idea": name, "alarm": al, "kw": kw, "seed": seed, "days": rec, "periods": P})
            OUT.write_text(json.dumps(res))
            rel = 100 * sum(r["released"] for r in rec) / sum(r["fc"] for r in rec)
            print(name, seed, [round(p["missed_pct"], 2) for p in P], "released %.2f" % rel, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
