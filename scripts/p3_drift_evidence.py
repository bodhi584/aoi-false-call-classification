"""P3: evidence of drift in the whole data set (all 19 weeks, 440,274 rows).

    python scripts/p3_drift_evidence.py   ->  results/p3_drift.json

Per week (7 calendar days from the first day):
  share_pct            true defects in % of the flagged joints            -> does p(y) change?
  inspection_type_share_pct  share of the rows of each inspection type           -> does p(X) change? (simple view)
  auroc_vs_weeks_1_4   AUROC of a random forest that separates the rows of the week from the rows of
                       weeks 1-4, from the inputs only (no class); 0.5 = same distribution  -> does p(X) change?
Concept drift is a change of p(X, y) over time [12, Sect. 2.1, Eq. 2].
"""
import json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
df = pd.read_csv("data/dataset.csv", index_col=0)
ts = pd.to_datetime(df["timestamp"]); day = (ts.dt.normalize() - ts.dt.normalize().min()).dt.days
week = (day // 7 + 1).to_numpy(); y = df["class"].to_numpy()
X = df.drop(columns=["timestamp", "class"]); X = pd.get_dummies(X, columns=[c for c in X.columns if not c.startswith("inspection_feat")]).to_numpy(dtype=np.float32)
print(X.shape, week.max(), len(np.unique(day)))
out = {"weeks": [], "rows": [], "true_defects": [], "share_pct": [], "auroc_vs_weeks_1_4": []}
rng = np.random.default_rng(0)
ref = np.flatnonzero(week <= 4)
def auc(a, b, n=6000):
    a = rng.choice(a, min(n, len(a)), replace=False); b = rng.choice(b, min(n, len(b)), replace=False)
    Xa = np.vstack([X[a], X[b]]); t = np.r_[np.zeros(len(a)), np.ones(len(b))]
    p = cross_val_predict(RandomForestClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=0), Xa, t, cv=StratifiedKFold(3, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return float(roc_auc_score(t, p))
h = rng.permutation(ref); base = auc(h[: len(h) // 2], h[len(h) // 2:])
for w in range(1, week.max() + 1):
    i = np.flatnonzero(week == w)
    out["weeks"].append(w); out["rows"].append(int(len(i))); out["true_defects"].append(int(y[i].sum())); out["share_pct"].append(round(100 * y[i].mean(), 3))
    out["auroc_vs_weeks_1_4"].append(None if w <= 4 else round(auc(ref, i), 3))
out["auroc_weeks_1_4_random_halves"] = round(base, 3)
t = (100 * pd.crosstab(week, df["inspection_type"].to_numpy(), normalize="index")).round(1)
out["inspection_type_share_pct"] = {str(c): [float(v) for v in t[c]] for c in t.columns}
out["note"] = "week = 7 calendar days from the first day; AUROC of a random forest (200 trees, 3-fold CV, up to 6,000 rows per side, inputs only, no class) that separates the rows of a week from the rows of weeks 1-4; 0.5 = same input distribution; inspection_type_share_pct: share of the rows of each inspection type per week"
json.dump(out, open("results/p3_drift.json", "w"), indent=1)
print(json.dumps(out))
