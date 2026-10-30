"""Robustness check of the classifier screening: does the ranking hold on other time windows?

Rolling origin (walk-forward): train on all days before the origin, test on the next 20 days.
Three origins: day 40, 60 and 80 of 132. Candidates: the best member of each family from
compare_classifiers.py that can be trained on the full data.

Usage: python scripts/compare_classifiers_folds.py [data/dataset.csv]   (about 15 minutes)
Output: results/classifier_screening_folds.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aoi_classification.data import load  # noqa: E402
from compare_classifiers import CANDIDATES, evaluate  # noqa: E402

KEEP = ["logistic_regression", "gaussian_naive_bayes", "random_forest", "extra_trees",
        "hist_gradient_boosting", "mlp_1_hidden_layer"]
ORIGINS = [40, 60, 80]
TEST_DAYS = 20


def main(path="data/dataset.csv"):
    df = load(path)
    make = {name: m for _, name, m, _ in CANDIDATES}
    res = {}
    for name in KEEP:
        res[name] = []
        for o in ORIGINS:
            r = evaluate(df[df.day < o], df[(df.day >= o) & (df.day < o + TEST_DAYS)], make[name])
            res[name].append({"origin_day": o, **r})
            print(name, o, json.dumps(r), flush=True)
        Path("results/classifier_screening_folds.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
