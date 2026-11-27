"""P4: rolling simulation of the AOI second check on the development half (days 1-64).

One step at time T (slide 3a of the P3 presentation):
  train (4): all labelled joints before T, except the threshold set  -> balanced random forest (BRF)
  t (4):     threshold set = the latest m labelled true defects before T, never used for training;
             t = the k-th lowest BRF score of them, k from the Neyman-Pearson rule [14, Alg. 1]
  check (5): the joints from T to the next retraining; released if s(x) < t
Then T moves on by one retraining interval.

Labels: before the start T0 every joint has its manual label. After T0 an operator labels every
joint that is not released; of the released joints only a random audit share a is labelled.
"""
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from scipy.stats import binom
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4", "inspection_type"]
ALPHA, DELTA = 0.01, 0.05          # at most 1 % of true defects released, with 95 % confidence
MIN_TRAIN_TD = 50                  # fewer true defects in the training window: keep the old model


# ---------------------------------------------------------------- data preparation
def load(path):
    """Rows in time order. Returns X (206 inputs), y (1 = true defect), calendar day of each row, n_dev."""
    df = pd.read_csv(path, index_col=0)
    date = pd.to_datetime(df["timestamp"]).dt.normalize()
    cal_day = ((date - date.iloc[0]).dt.days).to_numpy()            # 0 = first production day
    num = df.drop(columns=CATEGORICAL + ["class", "timestamp"])     # 70 measurements, used as they are
    # 5 categorical features -> one 0/1 column per category (136 columns); category codes are product and
    # component IDs, so the full code list is taken from the whole file (it holds no label information)
    enc = OneHotEncoder(sparse_output=False, dtype=np.float32).fit(df[CATEGORICAL])
    X = np.hstack([num.to_numpy(np.float32), enc.transform(df[CATEGORICAL])])   # 70 + 136 = 206 inputs
    y = df["class"].to_numpy()
    n_dev = len(df) // 2                                            # first half of the rows = development
    return X, y, cal_day, n_dev


# ---------------------------------------------------------------- threshold rules
def np_rank(m, alpha=ALPHA, delta=DELTA):
    """Largest rank k with P(fewer than k of m true-defect scores fall in the lowest alpha) <= delta.

    [14, Alg. 1], mirrored to the low end: releasing below the k-th lowest score misses more than
    alpha only if fewer than k of the m scores lie in the lowest alpha of all true-defect scores,
    and that count is binomial(m, alpha). Returns 0 if m is too small (m < 299): nothing is released.
    """
    k = 0
    while binom.cdf(k, m, alpha) <= delta:      # cdf(k) = P(at most k) = P(fewer than k + 1)
        k += 1
    return k                                     # m=299 -> 1, m=500 -> 2, m=1000 -> 5


def plain_rank(m, alpha=ALPHA):
    """Plain empirical quantile: release below the score that has alpha*m scores under it."""
    return int(np.floor(alpha * m)) + 1          # m=1000 -> 11 (10 below it = 1 %)


def threshold(scores_td, rule):
    """t from the scores of the threshold set; release if s(x) < t."""
    s = np.sort(scores_td)
    k = np_rank(len(s)) if rule == "np" else plain_rank(len(s))
    return -np.inf if k == 0 else float(s[k - 1])   # -inf: release nothing


# ---------------------------------------------------------------- one run
FOREST_DEFAULTS = {"n_estimators": 100, "max_features": "sqrt", "min_samples_leaf": 1, "sampling_strategy": "all"}


def make_forest(seed, forest=None):
    """BRF, F1-F4 = library defaults of imbalanced-learn 0.14 (P3 slide 10) unless given; fixed seed.
    F1 n_estimators, F2 max_features, F3 min_samples_leaf, F4 sampling_strategy ("all" = 1:1;
    a number r = true defects : false calls after under-sampling, e.g. 0.5 = twice as many false calls)."""
    kw = FOREST_DEFAULTS | (forest or {})
    return BalancedRandomForestClassifier(random_state=seed, n_jobs=-1, **kw)


def cv_threshold(X, y, seed, recall=0.99):
    """Threshold method of [7]: 5-fold stratified CV; per fold the highest t keeping recall >= 99 %; mean."""
    from sklearn.model_selection import StratifiedKFold
    ts = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        s = make_forest(seed).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        pos = np.sort(s[y[va] == 1])[::-1]                          # true-defect scores, high to low
        ts.append(float(pos[int(np.ceil(recall * len(pos))) - 1]))  # 99 % of them stay at or above t
    return float(np.mean(ts))


def capped_latest(td_idx, cal_day, m, cap):
    """Latest m true defects, but at most cap from one calendar day (spreads the set over more days)."""
    if cap is None:
        return td_idx[-m:]
    out, per_day = [], {}
    for i in td_idx[::-1]:
        d = cal_day[i]
        if per_day.get(d, 0) < cap:
            out.append(i); per_day[d] = per_day.get(d, 0) + 1
            if len(out) == m:
                break
    return np.sort(np.array(out, dtype=int))


def load_groups(path, column="inspection_type"):
    """Group label of each row (for per-group thresholds)."""
    return pd.read_csv(path, usecols=[column])[column].to_numpy()


def group_thresholds(model, X, y_idx_td, groups, cal_day, m, cap, rule):
    """One threshold per group, each from the latest m true defects of that group (NP or plain rank).
    A group with fewer than 299 true defects gets k = 0: none of its joints is released."""
    out = {}
    for g in np.unique(groups):
        td_g = y_idx_td[groups[y_idx_td] == g]
        thr_set = capped_latest(td_g, cal_day, min(m, len(td_g)), cap) if len(td_g) else td_g
        out[g] = threshold(model.predict_proba(X[thr_set])[:, 1], rule) if len(thr_set) else -np.inf
    return out


def simulate(X, y, cal_day, n_dev, *, start_day, interval, window, m, rule, audit, seed, cap=None,
             groups=None, n_end=None, forest=None):
    """Rolling run on the development half. Returns one record per check window.

    start_day: calendar day of T0 (first deployment day)
    interval:  retraining interval in calendar days; None = train once at T0 (no retraining)
    window:    training window in calendar days before T; None = all history
    m:         size of the threshold set (true defects)
    rule:      "np" (Neyman-Pearson), "plain" (1 % quantile), "cv" (method of [7]), "half" (s >= 0.5 = defect)
    cap:       at most cap threshold-set true defects from one calendar day (None = no limit)
    groups:    group label per row (e.g. inspection type): one threshold per group; None = one for all
    n_end:     last row + 1 that is simulated (default n_dev); history before start_day is always labelled
    forest:    forest parameters F1-F4 that differ from the defaults (dict)
    audit:     share of released joints that the operator still labels (random audit)
    """
    rng = np.random.default_rng(seed)
    n_dev = n_end or n_dev                        # rows that take part (development half by default)
    idx = np.arange(n_dev)
    known = cal_day[:n_dev] < start_day           # before T0 every label is known
    step = interval or 7                          # without retraining, report in 7-day windows
    last = cal_day[n_dev - 1]
    records, model, t, skipped = [], None, None, False
    T = start_day
    while T <= last:
        if not records or interval is not None:                   # (re)train at T
            hist = idx[(cal_day[:n_dev] < T) & known]              # labelled history before T
            td = hist[y[hist] == 1]
            # threshold set = latest m labelled true defects (only for the rules that need one)
            thr_set = (capped_latest(td, cal_day, m, cap) if rule in ("np", "plain") and groups is None
                       else td[:0])
            if groups is not None:                                 # per group: leave out each group's latest m
                thr_set = np.concatenate([capped_latest(td[groups[td] == g], cal_day, m, cap)
                                          for g in np.unique(groups) if (groups[td] == g).any()]).astype(int)
            train = np.setdiff1d(hist, thr_set)
            if window is not None:
                train = train[cal_day[train] >= T - window]
            n_train_td = int(y[train].sum())
            if n_train_td < MIN_TRAIN_TD:                          # window too short for the forest
                if model is None:                                  # no model yet: release nothing
                    model, t = None, -np.inf
                skipped = True
            else:
                model, skipped = make_forest(seed, forest).fit(X[train], y[train]), False
            if skipped:
                pass
            elif groups is not None:
                t = group_thresholds(model, X, td, groups, cal_day, m, cap, rule)
            elif rule in ("np", "plain"):
                t = threshold(model.predict_proba(X[thr_set])[:, 1], rule)
            elif rule == "cv":
                t = cv_threshold(X[train], y[train], seed)
            else:
                t = 0.5                                            # classical: defect if s(x) >= 0.5
        chk = idx[(cal_day[:n_dev] >= T) & (cal_day[:n_dev] < T + step)]
        if len(chk):
            if model is None:
                rel = np.zeros(len(chk), bool)
            else:
                sc = model.predict_proba(X[chk])[:, 1]
                tt = np.array([t.get(g, -np.inf) for g in groups[chk]]) if isinstance(t, dict) else t
                rel = sc < tt
            yc = y[chk]
            records.append({"T": int(T), "td": int(yc.sum()), "fc": int((yc == 0).sum()),
                            "missed": int((rel & (yc == 1)).sum()), "released": int((rel & (yc == 0)).sum()),
                            "t": ({str(k): float(v) for k, v in t.items()} if isinstance(t, dict) else float(t)), "train_td": n_train_td, "kept_old_model": bool(skipped)})
            # labels that come back: every operator-checked joint, plus the audited share of released ones
            known[chk[~rel]] = True
            known[chk[rel]] = rng.random(int(rel.sum())) < audit
        T += step
    return records


def periods(records, min_td=299):
    """Pool consecutive check windows into periods of at least min_td true defects (rest joins the last)."""
    out, cur = [], None
    for r in records:
        cur = {k: 0 for k in ("td", "fc", "missed", "released")} if cur is None else cur
        for k in cur:
            cur[k] += r[k]
        if cur["td"] >= min_td:
            out.append(cur); cur = None
    if cur is not None:
        if out:
            for k in cur:
                out[-1][k] += cur[k]
        else:
            out.append(cur)
    for p in out:
        p["missed_pct"] = 100 * p["missed"] / max(p["td"], 1)
        p["released_pct"] = 100 * p["released"] / max(p["fc"], 1)
    return out
