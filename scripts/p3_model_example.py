"""P3: one real example of the model, for the slides that explain the balanced random forest (BRF).

    python scripts/p3_model_example.py   ->  results/p3_model_example.json

The model is the one of setting "once" for seed 42 (method of [7] on our split): fitted on a stratified
80 % of the training days 1-22, with the hyper-parameters [7] logged for seed 42 (data/runs.csv).
The script reports, with real numbers:
    data       rows and true defects before and after the false calls are thinned out
    tree 1     the first two levels: input, cut value, group size, true-defect share, Gini impurity
    root       the Gini calculation of the first split of tree 1 (parent, two children, gain)
    forest     depth and leaves of all trees
    votes      tree votes and score s for three true defects and three false calls of period P1
    threshold  the CV threshold of [7] for this model (with its 5 fold thresholds), and the q = 1 % threshold
               of feature 2
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent))
from p3_pipeline import MODEL, TRAIN_DAYS, data, past_threshold, periods, seed_params  # noqa: E402
from replicate_pfab2025 import CATEGORICAL, cv_threshold, fit, threshold_at_recall  # noqa: E402

OUT = Path("results/p3_model_example.json")


def input_names():
    """Names of the 206 model inputs, in the column order of load(): numbers first, then one-hot categories."""
    df = pd.read_csv("data/dataset.csv", index_col=0).drop(columns="timestamp")
    enc = OneHotEncoder(sparse_output=False).fit(df[CATEGORICAL])
    num = [c for c in df.columns if c not in CATEGORICAL + ["class"]]
    return num + [f"{c}={v}" for c, cats in zip(CATEGORICAL, enc.categories_) for v in cats]


def main():
    X, y, days, h = data()
    p, seed = seed_params()[0]                                     # seed 42, first logged run of [7]
    idx_fit, _ = train_test_split(np.arange(h), test_size=0.2, stratify=y[:h], random_state=seed)
    model = fit(MODEL, p, seed, X[idx_fit], y[idx_fit])            # same call as setting "once"
    sc, rf = model.named_steps["scaler"], model.named_steps["model"]
    names = input_names()
    out = {"seed": seed, "train_days": TRAIN_DAYS, "rows_training_part": int(h), "true_defects_training_part": int(y[:h].sum()),
           "rows_fit": int(len(idx_fit)), "true_defects_fit": int(y[idx_fit].sum()), "inputs": int(X.shape[1]),
           "trees": len(rf.estimators_), "max_features": str(rf.max_features), "criterion": rf.criterion,
           "class_weight": str(rf.class_weight), "sampling_ratio": float(p["sampling_ratio"]), "scaler": type(sc).__name__}

    # step 1 of fit(): false calls thinned to true defects : false calls = 1 : sampling_ratio
    _, ys = RandomUnderSampler(sampling_strategy=1 / p["sampling_ratio"], random_state=seed).fit_resample(X[idx_fit], y[idx_fit])
    out["rows_after_thinning"], out["true_defects_after_thinning"] = int(len(ys)), int(ys.sum())

    def unscale(f, thr):                                           # cut value in the units of the raw data
        v = np.zeros((1, X.shape[1])); v[0, f] = thr
        return float(sc.inverse_transform(v)[0, f])

    trees = [e.steps[-1][1] if hasattr(e, "steps") else e for e in rf.estimators_]
    t = trees[0].tree_

    def node(n):
        v = t.value[n][0]
        d = {"rows_drawn": int(t.n_node_samples[n]), "weighted": float(t.weighted_n_node_samples[n]),
             "true_defect_share": float(v[1] / v.sum()), "gini": float(t.impurity[n])}
        if t.children_left[n] != -1:
            d.update(input=names[t.feature[n]], cut=unscale(t.feature[n], t.threshold[n]),
                     yes=int(t.children_left[n]), no=int(t.children_right[n]))
        return d

    lv = {0: node(0)}
    for k in [t.children_left[0], t.children_right[0]]:
        lv[int(k)] = node(k)
        if t.children_left[k] != -1:
            for kk in [t.children_left[k], t.children_right[k]]:
                lv[int(kk)] = node(kk)
    out["tree1"] = {str(k): v for k, v in lv.items()}
    out["tree1_depth"], out["tree1_leaves"] = int(trees[0].get_depth()), int(trees[0].get_n_leaves())
    dep = [e.get_depth() for e in trees]; lea = [e.get_n_leaves() for e in trees]
    out["forest_depth_min_max"], out["forest_leaves_min_max"] = [int(min(dep)), int(max(dep))], [int(min(lea)), int(max(lea))]

    # Gini of the root split: G = 1 - p^2 - (1 - p)^2, gain = G(parent) - weighted G(children)
    L, R = t.children_left[0], t.children_right[0]
    wl, wr = t.weighted_n_node_samples[L], t.weighted_n_node_samples[R]
    out["root_gain"] = float(t.impurity[0] - (wl * t.impurity[L] + wr * t.impurity[R]) / (wl + wr))

    # votes for real joints of period P1 (rows after day 22): each tree answers 0 or 1 (its leaves are pure),
    # the score s is the share of trees that answer 1. Examples: the first true defect of P1, the first false
    # call of P1 that is released at the CV threshold of [7], and the first one that is not.
    a, b = periods(len(y), h)[:2]
    Xp = sc.transform(X[a:b]); yp = y[a:b]
    s = model.predict_proba(X[a:b])[:, 1]
    out["cv_threshold"] = cv_threshold(MODEL, p, seed, X[idx_fit], y[idx_fit])
    # how [7] gets it: 5 folds of the fit set, ~36 true defects each; the 99 % recall threshold of a fold
    # with fewer than 100 true defects is its lowest true-defect score; the CV threshold is their mean
    folds = []
    Xf, yf = X[idx_fit], y[idx_fit]
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xf, yf):
        sv = fit(MODEL, p, seed, Xf[tr], yf[tr]).predict_proba(Xf[va])[:, 1]
        folds.append({"true_defects": int(yf[va].sum()), "threshold_99_recall": threshold_at_recall(yf[va], sv),
                      "lowest_true_defect_score": float(sv[yf[va] == 1].min())})
    out["cv_folds"] = folds
    t_cv = out["cv_threshold"]
    picks = [("true defect", np.flatnonzero(yp == 1)[0]),
             ("false call, released", np.flatnonzero((yp == 0) & (s < t_cv))[0]),
             ("false call, to manual review", np.flatnonzero((yp == 0) & (s >= t_cv))[0])]
    votes = []
    for lab, i in picks:
        pt = np.array([e.predict_proba(Xp[i:i + 1])[0, 1] for e in trees])
        votes.append({"row": int(a + i), "label": lab, "score": float(s[i]), "trees_voting_defect": int((pt > 0.5).sum()),
                      "tree_answers_are_0_or_1": bool(np.isin(pt, [0, 1]).all())})
    out["votes_P1"] = votes
    out["P1_rows"], out["P1_true_defects"] = int(b - a), int(yp.sum())

    # feature 2 at the start: q = 1 % threshold from the out-of-sample scores of the 223 true defects of days 1-22
    past = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.arange(h), y[:h]):
        mk = fit(MODEL, p, seed, X[tr], y[tr])
        past.extend(mk.predict_proba(X[te[y[te] == 1]])[:, 1])
    srt = np.sort(past)
    out["past_scores_start"] = len(past)
    out["past_lowest_5"] = [float(v) for v in srt[:5]]
    out["q1_threshold_start"] = past_threshold(past, 0.01)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
