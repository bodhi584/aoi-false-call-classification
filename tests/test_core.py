"""Code tests R1-R3 of docs/test_plan.md on the method of [7] (scripts/replicate_pfab2025.py), made-up data."""
import numpy as np
from sklearn.metrics import recall_score

from replicate_pfab2025 import cv_threshold, fit, split, threshold_at_recall


def test_r1_split_as_in_the_paper(data):
    """R1: first half 80/20 stratified, second half 5 periods in time order, no row in two parts."""
    X, y = data
    idx = np.arange(len(y)).reshape(-1, 1)            # use the row number as the only feature
    X_hp, y_hp, X_test, y_test, slices = split(idx, y, seed=42)
    first = np.concatenate([X_hp[:, 0], X_test[:, 0]])
    assert len(X_test) == 200 and len(X_hp) == 800    # 80/20 of the first 1,000 rows
    assert abs(y_test.mean() - y[:1000].mean()) < 0.01  # stratified
    assert set(first) == set(range(1000))              # first half only
    later = np.concatenate([s[0][:, 0] for s in slices])
    assert list(later) == list(range(1000, 2000))      # 5 periods in time order
    assert len(set(first) | set(later)) == 2000        # no row in two parts


def test_r2_threshold_keeps_99_percent_recall():
    """R2: the threshold keeps recall >= 99 % on the data it is set on."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = (rng.random(3000) < 0.05).astype(int)
        s = rng.random(3000) + 0.3 * y
        t = threshold_at_recall(y, s)
        assert recall_score(y, (s >= t).astype(int)) >= 0.99


def test_r3_threshold_belongs_to_the_final_model(data, params):
    """R3: the CV threshold is computed on the final training rows, not taken from the run log of [7]."""
    X, y = data
    t = cv_threshold("RFC", params, 42, X[:1000], y[:1000])
    assert t != params["SetThreshold"]
    m = fit("RFC", params, 42, X[:1000], y[:1000])
    assert recall_score(y[:1000], (m.predict_proba(X[:1000])[:, 1] >= t).astype(int)) > 0.9
