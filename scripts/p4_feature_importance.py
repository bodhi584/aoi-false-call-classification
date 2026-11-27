"""Which features matter for the true-defect decision? Permutation importance of the BRF (P4).

Development half only (days 1-64), in time order: BRF fitted on the first 80 % of its rows, importance
measured on the last 20 %. Score = average precision (area under the precision-recall curve), which suits
the rare class. Each of the 75 features is permuted as a whole (a categorical feature = all its 0/1
columns together), 5 repeats; importance = mean drop of the score. Done for all rows and per inspection
type, because one inspection_feat column holds a different physical quantity per inspection type [4, Sect. 2].
Output: results/p4_feature_importance.json, docs/img/p4/feature_importance.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import CATEGORICAL, make_forest  # noqa: E402

NAVY, BLUE = "#14213D", "#2F6DB5"


def main(path="data/dataset.csv", repeats=5, seed=0):
    df = pd.read_csv(path, index_col=0)
    dev = df.iloc[: len(df) // 2]                           # development half only
    num = [c for c in dev.columns if c.startswith("inspection_feat")]
    X = pd.get_dummies(dev[num + CATEGORICAL].astype({c: "category" for c in CATEGORICAL}),
                       columns=CATEGORICAL, dtype=np.float32)
    groups = {c: [c] for c in num} | {c: [k for k in X.columns if k.startswith(c + "_")] for c in CATEGORICAL}
    y = dev["class"].to_numpy()
    cut = int(0.8 * len(dev))                               # time order: first 80 % fit, last 20 % measure
    model = make_forest(seed).fit(X.iloc[:cut], y[:cut])
    Xv, yv, itype = X.iloc[cut:].reset_index(drop=True), y[cut:], dev["inspection_type"].to_numpy()[cut:]
    rng = np.random.default_rng(seed)
    base = model.predict_proba(Xv)[:, 1]

    def drop(mask):
        """Mean drop of average precision on the rows in mask when each feature group is permuted."""
        ref = average_precision_score(yv[mask], base[mask])
        out = {}
        for g, cols in groups.items():
            d = []
            for _ in range(repeats):
                Xp = Xv.copy()
                Xp[cols] = Xp[cols].to_numpy()[rng.permutation(len(Xp))]
                d.append(ref - average_precision_score(yv[mask], model.predict_proba(Xp)[:, 1][mask]))
            out[g] = float(np.mean(d))
        return ref, out

    res = {"rows_fit": cut, "rows_measure": len(yv), "true_defects_measure": int(yv.sum())}
    ref, imp = drop(np.ones(len(yv), bool))
    res["all"] = {"average_precision": ref, "importance": dict(sorted(imp.items(), key=lambda x: -x[1]))}
    for t in sorted(np.unique(itype)):
        mask = itype == t
        if yv[mask].sum() >= 20:                            # enough true defects to measure
            r, i = drop(mask)
            res[f"type_{t}"] = {"average_precision": r, "true_defects": int(yv[mask].sum()),
                                "importance": dict(sorted(i.items(), key=lambda x: -x[1])[:10])}
    Path("results").mkdir(exist_ok=True)
    Path("results/p4_feature_importance.json").write_text(json.dumps(res, indent=2))

    top = list(res["all"]["importance"].items())[:15][::-1]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.barh([k for k, _ in top], [v for _, v in top],
            color=[BLUE if k in CATEGORICAL else NAVY for k, _ in top])
    ax.set_xlabel("drop of average precision when the feature is permuted")
    ax.set_title("BRF permutation importance, top 15 of 75 features (development half)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    Path("docs/img/p4").mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig("docs/img/p4/feature_importance.png")
    print(json.dumps({k: v for k, v in list(res["all"]["importance"].items())[:10]}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
