"""P4 experiment 7 (added 28 Nov, after the final test, development half only):
are the model choice (BRF) and the confidence 95 % (S4) supported by our own data?

Same system as the frozen model (every 14 d, all history, m = 500, audit 10 %, NP), only the model changes:
BRF (frozen), random forest (RF), weighted random forest (RF with class_weight balanced_subsample = WRF),
XGBoost (scale_pos_weight balanced or 1), logistic regression (balanced), neural network (MLP on a 1:1 under-sample).
Each model gets its library defaults and one variant. Confidence: NP 99 % with the frozen BRF
(NP 90 % gives the same rank k = 2 as 95 % for m = 500, so it is identical to the frozen model).
5 seeds. Output: results/p4_dev_models.json. This does not change the reported final-test result.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load, periods, simulate  # noqa: E402

BRF = {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "sampling_strategy": "all"}
MODELS = {
    "BRF (frozen)": (BRF, "np"),
    "BRF, NP 99 %": (BRF, "np99"),
    "RF, 300 trees, leaf 1": ({"model": "rf", "n_estimators": 300}, "np"),
    "RF, 300 trees, leaf 5": ({"model": "rf", "n_estimators": 300, "min_samples_leaf": 5}, "np"),
    "WRF, 300 trees, leaf 1": ({"model": "rf", "n_estimators": 300, "class_weight": "balanced_subsample"}, "np"),
    "WRF, 300 trees, leaf 5": ({"model": "rf", "n_estimators": 300, "min_samples_leaf": 5,
                                "class_weight": "balanced_subsample"}, "np"),
    "XGBoost, balanced weights": ({"model": "xgb", "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1}, "np"),
    "XGBoost, no weights": ({"model": "xgb", "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
                             "scale_pos_weight": 1}, "np"),
    "Logistic regression": ({"model": "logreg"}, "np"),
    "Logistic regression, C 0.1": ({"model": "logreg", "C": 0.1}, "np"),
    "Neural network 64": ({"model": "mlp", "hidden_layer_sizes": (64,)}, "np"),
    "Neural network 128-64": ({"model": "mlp", "hidden_layer_sizes": (128, 64)}, "np"),
}
SYSTEM = {"interval": 14, "window": None, "m": 500, "audit": 0.1}
OUT = Path("results/p4_dev_models.json")
ORDER = ["BRF (frozen)", "BRF, NP 99 %", "XGBoost, balanced weights", "XGBoost, no weights", "Logistic regression",
         "Logistic regression, C 0.1", "Neural network 64", "Neural network 128-64",
         "RF, 300 trees, leaf 1", "RF, 300 trees, leaf 5", "WRF, 300 trees, leaf 1", "WRF, 300 trees, leaf 5"]


def main(path="data/dataset.csv"):
    X, y, cal_day, n_dev = load(path)
    start = json.loads(Path("results/p4_dev_runs.json").read_text())["start_day"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {"start_day": start, "system": SYSTEM, "runs": []}
    done = {(r["model"], r["seed"]) for r in res["runs"]}
    for name in ORDER:
        forest, rule = MODELS[name]
        for seed in range(5):
            if (name, seed) in done:
                continue
            rec = simulate(X, y, cal_day, n_dev, start_day=start, seed=seed, rule=rule, forest=forest, **SYSTEM)
            P = periods(rec)
            res["runs"].append({"model": name, "forest": {k: list(v) if isinstance(v, tuple) else v
                                                          for k, v in forest.items()},
                                "rule": rule, "seed": seed, "windows": rec, "periods": P})
            OUT.write_text(json.dumps(res))
            rel = 100 * sum(w["released"] for w in rec) / sum(w["fc"] for w in rec)
            print(name, seed, [round(p["missed_pct"], 2) for p in P], "released %.2f" % rel, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
