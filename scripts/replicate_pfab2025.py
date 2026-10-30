"""One-to-one replication of Pfab & Rothering (2025), with one change: retraining.

Reference: K. Pfab and M. Rothering, "Towards improved research methodologies for industrial AI:
a case study of false call reduction", arXiv:2506.14521, 2025. Their source code
(https://gitlab.com/research9841542/FCR_1) and their logged runs were read; this file is our own
implementation of the protocol described there, so that only the retraining differs.

Protocol (same as the paper):
- Data: Siemens AOI data, one row = one flagged solder joint, in time order. Timestamp dropped,
  the 5 categorical columns one-hot encoded.
- Split by rows: first half -> stratified random 80 % "hyper-parameter set" / 20 % "test set";
  second half -> 5 chronological evaluation slices of equal size.
- Model per seed: the hyper-parameters, scaler and under-sampling ratio that the authors logged for
  that seed (requirement-aware run "ConstrainedAUC", adapted threshold).
- Threshold: 5-fold stratified cross-validation; in each fold the highest threshold that keeps
  recall >= 99 %; the mean of the 5 is used.
- Metrics on joint level: slip = 1 - recall (true defects released), volume reduction = recall of
  class 0 (false calls released). Targets: slip <= 1 %, volume reduction >= 40 %.

Policies:
- train_once (the paper): train on the hyper-parameter set once, use it for all 5 slices.
- retrain (ours): before each slice, retrain on all labelled data seen so far (hyper-parameter set,
  test set and earlier slices) with the same hyper-parameters, and set a new threshold with the same
  cross-validation method on that data.

Assumption (limitation): every joint gets its manual-inspection label later, also the released ones.

Usage: python scripts/replicate_pfab2025.py RUNS_CSV [data/dataset.csv]
Output: results/replication_pfab2025.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

TARGET_RECALL = 0.99
CATEGORICAL = ["meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4", "inspection_type"]
SCALERS = [StandardScaler, MinMaxScaler]


def load(path):
    """Rows in time order; one-hot categories; X, y as numpy arrays."""
    df = pd.read_csv(path, index_col=0).drop(columns="timestamp")
    enc = OneHotEncoder(sparse_output=False).fit(df[CATEGORICAL])
    X = np.hstack([df.drop(columns=CATEGORICAL + ["class"]).values, enc.transform(df[CATEGORICAL])])
    return X, df["class"].values


def split(X, y, seed):
    """The paper's split: first half -> 80/20 stratified random; second half -> 5 slices."""
    n = len(y)
    X1, y1, X2, y2 = X[: int(n * 0.5)], y[: int(n * 0.5)], X[int(n * 0.5):], y[int(n * 0.5):]
    X_hp, X_test, y_hp, y_test = train_test_split(X1, y1, test_size=0.2, stratify=y1, random_state=seed)
    m = len(y2)
    slices = [(X2[int(i * 0.2 * m): int((i + 1) * 0.2 * m)], y2[int(i * 0.2 * m): int((i + 1) * 0.2 * m)])
              for i in range(5)]
    return X_hp, y_hp, X_test, y_test, slices


def build(model, p, seed):
    """The authors' model with the hyper-parameters they logged for this seed."""
    crit_rf, crit_brf = ["gini", "entropy", "log_loss"], ["gini", "entropy"]
    feats = ["sqrt", "log2", None]
    weights = ["balanced_subsample", "balanced", None]
    kw = dict(n_estimators=int(p["n_estimators"]), max_features=feats[int(np.trunc(p["max_features"]))],
              class_weight=weights[int(np.trunc(p["class_weight"]))], random_state=seed, n_jobs=-1)
    if model == "RFC":
        clf = RandomForestClassifier(criterion=crit_rf[int(np.trunc(p["criterion"]))], **kw)
    else:
        clf = BalancedRandomForestClassifier(criterion=crit_brf[int(np.trunc(p["criterion"]))],
                                             sampling_strategy="auto", replacement=False, bootstrap=True, **kw)
    return Pipeline([("scaler", SCALERS[int(p["scaler_kind"])]()), ("model", clf)])


def fit(model, p, seed, X, y):
    """Under-sample the false calls as in the paper, then train."""
    Xs, ys = RandomUnderSampler(sampling_strategy=1 / p["sampling_ratio"], random_state=seed).fit_resample(X, y)
    return build(model, p, seed).fit(Xs, ys)


def threshold_at_recall(y, score):
    """Highest threshold with recall >= 99 % (same rule as the paper's get_proper_threshold)."""
    pos = np.sort(score[y == 1])[::-1]
    k = int(np.ceil(TARGET_RECALL * len(pos)))
    return float(pos[k - 1])


def cv_threshold(model, p, seed, X, y):
    """The paper's threshold method: mean of the 5 fold thresholds."""
    ts = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = fit(model, p, seed, X[tr], y[tr])
        ts.append(threshold_at_recall(y[va], m.predict_proba(X[va])[:, 1]))
    return float(np.mean(ts))


def scores(m, t, X, y):
    released = m.predict_proba(X)[:, 1] < t
    return {"slip": float(released[y == 1].mean()), "volume_reduction": float(released[y == 0].mean())}


def run(model, p, seed, X, y):
    X_hp, y_hp, X_test, y_test, slices = split(X, y, seed)
    out = {"seed": seed, "logged_threshold": float(p["SetThreshold"])}
    # train once (the paper)
    m = fit(model, p, seed, X_hp, y_hp)
    t = cv_threshold(model, p, seed, X_hp, y_hp)
    out["threshold_once"] = t
    out["test"] = scores(m, t, X_test, y_test)
    out["train_once"] = [scores(m, t, Xs, ys) for Xs, ys in slices]
    # retrain before every slice (ours)
    Xa, ya = np.vstack([X_hp, X_test]), np.concatenate([y_hp, y_test])
    res = []
    for Xs, ys in slices:
        m = fit(model, p, seed, Xa, ya)
        t = cv_threshold(model, p, seed, Xa, ya)
        res.append(scores(m, t, Xs, ys))
        Xa, ya = np.vstack([Xa, Xs]), np.concatenate([ya, ys])
    out["retrain"] = res
    return out


def main(runs_csv, path="data/dataset.csv"):
    runs = pd.read_csv(runs_csv)
    runs["base"] = runs["Name"].str.replace(r"_\d+$", "", regex=True)
    X, y = load(path)
    result = {}
    for model in ["RFC", "BRFC"]:
        sel = runs[(runs.base == f"{model}_ConstrainedAUC_AdaptedThreshold") & runs.n_estimators.notna()]
        result[model] = []
        for _, p in sel.iterrows():
            r = run(model, p, int(p["Seed"]), X, y)
            r["paper"] = {"test_slip": float(1 - p["RecallTest"]),
                          "slices_slip": [float(1 - p[f"RecallEval{i}"]) for i in range(5)],
                          "slices_volume_reduction": [float(p[f"VolumeReductionEval{i} "]) for i in range(5)]}
            result[model].append(r)
            print(model, r["seed"], "once", [round(s["slip"], 3) for s in r["train_once"]],
                  "retrain", [round(s["slip"], 3) for s in r["retrain"]],
                  "paper", [round(s, 3) for s in r["paper"]["slices_slip"]], flush=True)
            Path("results").mkdir(exist_ok=True)
            Path("results/replication_pfab2025.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
