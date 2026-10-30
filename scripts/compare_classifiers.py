"""Screening of classifier families for the second check (model selection, literature report section 2.2.3).

Every candidate gets the same task: train on the first 70 % of the days, score the remaining boards, and
report the specificity (false-call boards correctly classified) at 99 % and 100 % recall, joint AUROC and
the training time. Same time-order split as run_baseline.py. Thresholds are read on the test set, so the
numbers are an upper bound for every model alike; they are used only to compare the families.

Usage: python scripts/compare_classifiers.py [data/dataset.csv]   (about 30 minutes on 2 CPU cores)
Output: results/classifier_screening.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aoi_classification.data import CATEGORICAL, load  # noqa: E402
from aoi_classification.model import board_scores, threshold_for, to_X  # noqa: E402

NUM = make_column_selector(pattern="^inspection_feat")


def scaled():
    """Categories as 0/1 columns, measurements scaled: for linear, distance and neural models."""
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                              ("num", StandardScaler(), NUM)])


def ordinal():
    """Categories as integer codes, raw measurements: for tree models."""
    return ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
                              ("num", "passthrough", NUM)])


# family, name, pipeline, note. class_weight / balancing is used wherever the model supports it.
CANDIDATES = [
    ("linear", "logistic_regression", lambda: make_pipeline(scaled(), LogisticRegression(max_iter=1000, class_weight="balanced")), ""),
    ("linear", "linear_discriminant_analysis", lambda: make_pipeline(scaled(), LinearDiscriminantAnalysis()), ""),
    ("linear", "linear_svm", lambda: make_pipeline(scaled(), SGDClassifier(loss="hinge", class_weight="balanced", random_state=0)), "hinge loss, trained by SGD; decision_function score"),
    ("probabilistic", "gaussian_naive_bayes", lambda: make_pipeline(scaled(), GaussianNB()), ""),
    ("instance", "k_nearest_neighbours", None, "k=15; trained on a 50,000-row subsample: prediction cost grows with training size"),
    ("kernel", "rbf_svm_subsample", None, "trained on a 20,000-row subsample only: full data infeasible (quadratic cost)"),
    ("tree", "decision_tree", lambda: make_pipeline(ordinal(), DecisionTreeClassifier(min_samples_leaf=20, class_weight="balanced", random_state=0)), ""),
    ("tree_ensemble", "random_forest", lambda: make_pipeline(ordinal(), RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=0)), ""),
    ("tree_ensemble", "extra_trees", lambda: make_pipeline(ordinal(), ExtraTreesClassifier(n_estimators=100, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=0)), ""),
    ("boosting", "hist_gradient_boosting", lambda: make_pipeline(ordinal(), HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)), "scikit-learn's LightGBM-style GBDT"),
    ("boosting", "adaboost", lambda: make_pipeline(ordinal(), AdaBoostClassifier(n_estimators=200, random_state=0)), ""),
    ("neural", "mlp_1_hidden_layer", lambda: make_pipeline(scaled(), MLPClassifier(hidden_layer_sizes=(10,), early_stopping=True, max_iter=200, random_state=0)), "10 neurons"),
    ("neural", "mlp_2_hidden_layers", lambda: make_pipeline(scaled(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200, random_state=0)), "64+32 neurons"),
]


def score(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def evaluate(train, test, make):
    t0 = time.time()
    clf = make().fit(to_X(train), train["class"])
    fit_s = time.time() - t0
    p = score(clf, to_X(test))
    b = board_scores(test, p)
    out = {"joint_auroc": round(roc_auc_score(test["class"], p), 3), "fit_seconds": round(fit_s, 1)}
    for keep in (1.0, 0.99):
        thr = threshold_for(b.score.values, b.defective.values, 1 - keep)
        out[f"specificity_pct_at_{int(keep * 100)}pct_recall"] = round(
            100 * ((b.defective == 0) & (b.score < thr)).sum() / (b.defective == 0).sum(), 1)
    return out


def main(path="data/dataset.csv"):
    df = load(path)
    cut = df["day"].quantile(0.7)
    train, test = df[df.day <= cut], df[df.day > cut]
    out = Path("results/classifier_screening.json")
    done = json.loads(out.read_text()) if out.exists() else {}
    res = {}
    for family, name, make, note in CANDIDATES:
        if name in done:
            res[name] = done[name]
            continue
        if name == "k_nearest_neighbours":
            sub = train.sample(50000, random_state=0)
            r = evaluate(sub, test, lambda: make_pipeline(scaled(), KNeighborsClassifier(n_neighbors=15, n_jobs=-1)))
        elif name == "rbf_svm_subsample":
            sub = train.sample(20000, random_state=0)
            make = lambda: make_pipeline(scaled(), SVC(class_weight="balanced", probability=True, random_state=0))  # noqa: E731
            r = evaluate(sub, test, make)
        else:
            r = evaluate(train, test, make)
        res[name] = {"family": family, "note": note, **r}
        print(name, json.dumps(r), flush=True)
        Path("results").mkdir(exist_ok=True)
        Path("results/classifier_screening.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
