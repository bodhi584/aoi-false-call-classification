"""P3: the method of [7] and our two features, on the development data (production days 1-64) only.

    python scripts/p3_pipeline.py run      all settings x 10 seeds -> results/p3_results.json (resumable)
    python scripts/p3_pipeline.py report   table results/p3_summary.md and figures docs/img/p3_*.png
    python scripts/p3_pipeline.py check    re-run seed 1 of three settings and compare with the stored results

Data:     first half of data/dataset.csv = days 1-64 (220,137 rows), the half [7] uses for development.
          The later half (rest of day 64 to day 101) is cut off in dev_split() and kept for the final test in P4.
Split:    train on days 1-22, evaluate on days 23-64 in 5 periods of equal size, in time order.
Settings: once      method of [7]: train once, cross-validation (CV) threshold at 99 % recall
          period, weekly, 3days, daily   feature 1: retrain on all rows known so far, CV threshold each time
          q0 ... q2 feature 2: retrain weekly, threshold = q-quantile of past true-defect scores

Words used below:
    row / joint   one flagged solder joint = one row of data/dataset.csv
    true defect   class 1 (the person at the manual station confirmed a defect)
    false call    class 0 (the AOI flagged it, but the joint is good)
    score         model output between 0 and 1; high = looks like a true defect
    release       score below the threshold t -> the joint skips the manual station
    missed        a true defect that is released (slip in [7]); target <= 1 % in every period
    released      share of false calls that are released (volume reduction in [7]); as high as possible
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Functions of the method of [7], shared with the P2 replication (see comments there):
#   load()          read data/dataset.csv -> X (numbers), y (0/1), rows in time order
#   fit()           under-sample the false calls, then train the balanced random forest
#   cv_threshold()  mean of 5 fold thresholds, each the highest score threshold with recall >= 99 %
from replicate_pfab2025 import cv_threshold, fit, load

MODEL = "BRFC"        # balanced random forest, as in [7]
TRAIN_DAYS = 22       # training part: production days 1-22; evaluation part: days 23-64
SEEDS = 10            # the 10 runs [7] logged, each with its own seed and hyper-parameters
RESULTS = Path("results/p3_results.json")
# setting -> (retraining interval: None = never, "period" = before each period, or calendar days; q or None)
SETTINGS = {"once": (None, None), "period": ("period", None), "weekly": (7, None), "3days": (3, None),
            "daily": (1, None), "q0": (7, 0.0), "q0.1": (7, 0.001), "q0.2": (7, 0.002),
            "q0.5": (7, 0.005), "q1": (7, 0.01), "q2": (7, 0.02)}


# ---------- data ----------

def load_days():
    """Time of each row in days since midnight of the first record (rows are in time order)."""
    ts = pd.to_datetime(pd.read_csv("data/dataset.csv", index_col=0, usecols=[0, 1])["timestamp"])
    return ((ts - ts.iloc[0].floor("D")).dt.total_seconds() / 86400).values


def production_day(days):
    """Production day 1, 2, ... of each row; calendar days without production are not counted."""
    return np.unique(np.floor(days), return_inverse=True)[1] + 1


def dev_split(X, y, days, train_days):
    """Keep only the first half of the rows (days 1-64) and find the first row after production day
    `train_days`. Rows of the later half are cut off here, so no later step can use them (test V1)."""
    half = int(len(y) * 0.5)                               # first half by rows, as in [7]
    X, y, days = X[:half], y[:half], days[:half]           # rest of day 64 to day 101 dropped
    h = int(np.searchsorted(production_day(days), train_days + 1))   # first row of day 23
    return X, y, days, h


# ---------- feature 2: threshold from past true defects ----------

def past_threshold(past, q):
    """t = q-quantile of the scores that past true defects received from models that had not seen them:
    a share q of these scores lies below t. q = 0 gives the lowest past score (test M6)."""
    srt = np.sort(past)                                    # past true-defect scores, low to high
    return float(srt[int(np.floor(q * len(srt)))])        # the score at position q * count


def periods(n, h):
    """Borders of the 5 evaluation periods: rows h..n cut into 5 parts with the same number of rows.
    Returns 6 row numbers [start of P1, start of P2, ..., end of P5]."""
    m = n - h                                              # number of evaluation rows
    return [h + int(i * 0.2 * m) for i in range(6)]       # same rule as the slices of [7]


def retrain_starts(step, days, h, n):
    """First row of each retraining interval in the evaluation part; the model is retrained before each.
    step = "period" (before each of the 5 periods) or a number of calendar days (7 = weekly)."""
    if step == "period":
        return periods(n, h)                               # 5 period starts + end of the data
    k = np.floor((days[h:] - np.floor(days[h])) / step).astype(int)   # interval number 0, 1, 2 ... of each row
    new = np.flatnonzero(np.r_[True, np.diff(k) > 0])                 # positions where the number changes
    return [h + int(i) for i in new] + [n]                             # + end of the data as the last border


def run(p, seed, X, y, days, h, setting):
    """One run: returns, for each of the 5 evaluation periods, the share of missed true defects and of
    released false calls. p = hyper-parameters [7] logged for this seed (one row of data/runs.csv)."""
    step, q = SETTINGS[setting]                            # retraining interval and q of this setting
    n = len(y)                                             # rows of days 1-64 (the later half is already cut off)
    score = np.zeros(n)                                    # model score of every evaluation row
    thr = np.zeros(n)                                      # threshold in force for every evaluation row

    if step is None:
        # Method of [7]: 80 % of the training part to fit the model, chosen at random but with the same
        # share of true defects (stratified). The other 20 % are [7]'s random test set, not used here.
        idx_fit, _ = train_test_split(np.arange(h), test_size=0.2, stratify=y[:h], random_state=seed)
        model = fit(MODEL, p, seed, X[idx_fit], y[idx_fit])                 # train once
        t = cv_threshold(MODEL, p, seed, X[idx_fit], y[idx_fit])            # threshold once
        score[h:] = model.predict_proba(X[h:])[:, 1]                         # score all evaluation rows
        thr[h:] = t                                                          # same threshold for all of them

    else:
        starts = retrain_starts(step, days, h, n)          # borders of the retraining intervals
        if q is not None:
            # Feature 2, start: scores of the true defects of the training part from models that did NOT
            # see them. 5-fold: train on 4/5, score the true defects of the remaining 1/5, five times.
            past = []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.arange(h), y[:h]):
                mk = fit(MODEL, p, seed, X[tr], y[tr])                       # model without fold te
                te_defects = te[y[te] == 1]                                  # true defects of fold te
                past.extend(mk.predict_proba(X[te_defects])[:, 1])         # their out-of-sample scores

        for a, b in zip(starts[:-1], starts[1:]):          # one interval: rows a .. b-1
            known = np.arange(a)                           # all rows before it: their labels are known (test R4)
            model = fit(MODEL, p, seed, X[known], y[known])                  # feature 1: retrain on them
            if q is None:
                t = cv_threshold(MODEL, p, seed, X[known], y[known])         # threshold as in [7]
            else:
                t = past_threshold(past, q)                                  # feature 2: q-quantile of past scores
            score[a:b] = model.predict_proba(X[a:b])[:, 1]                   # score this interval
            thr[a:b] = t
            if q is not None:
                # After the week the review results are known: the true defects of this week join the
                # past scores. Their scores came from this week's model, which had not seen them.
                past.extend(score[a:b][y[a:b] == 1])

    released = score < thr                                 # below the threshold -> skip the manual station
    out = []
    for a, b in zip(periods(n, h)[:-1], periods(n, h)[1:]):
        r, yy = released[a:b], y[a:b]
        out.append({"missed": float(r[yy == 1].mean()),     # released true defects / all true defects
                    "released": float(r[yy == 0].mean())})  # released false calls / all false calls
    return out


def data():
    """Days 1-64 only, and the first row of the evaluation part (day 23)."""
    X, y = load("data/dataset.csv")                        # all 440,274 rows in time order
    return dev_split(X, y, load_days(), TRAIN_DAYS)        # cut off the later half; h = first row of day 23


def seed_params():
    """The 10 BRF runs [7] logged in data/runs.csv: hyper-parameters (n_estimators, max_features, ...) and seed."""
    runs = pd.read_csv("data/runs.csv")
    runs = runs[runs["Name"].str.startswith("BRFC_ConstrainedAUC_AdaptedThreshold") & runs.n_estimators.notna()]
    return [(p, int(p["Seed"])) for _, p in runs.head(SEEDS).iterrows()]


def run_all(only=None):
    """All settings x 10 seeds; the result file is saved after every run, finished runs are skipped."""
    X, y, days, h = data()
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for setting in [only] if only else SETTINGS:
        done = res.setdefault(setting, {})
        for p, seed in seed_params():
            if str(seed) in done:
                continue
            done[str(seed)] = run(p, seed, X, y, days, h, setting)
            RESULTS.write_text(json.dumps(res, indent=1))
            print(setting, seed, [round(r["missed"] * 100, 2) for r in done[str(seed)]], flush=True)


def table(setting, res):
    """Arrays seeds x 5 periods of missed and released shares in %, or None if not run."""
    if setting not in res or not res[setting]:
        return None
    runs = list(res[setting].values())
    missed = np.array([[r["missed"] for r in run] for run in runs]) * 100
    released = np.array([[r["released"] for r in run] for run in runs]) * 100
    return missed, released


NAMES = {"once": "Method of [7]: trained once, CV threshold", "period": "Retraining per period",
         "weekly": "Retraining weekly", "3days": "Retraining every 3 days", "daily": "Retraining daily",
         "q0": "Weekly + threshold q = 0", "q0.1": "Weekly + threshold q = 0.1 %", "q0.2": "Weekly + threshold q = 0.2 %",
         "q0.5": "Weekly + threshold q = 0.5 %", "q1": "Weekly + threshold q = 1 %", "q2": "Weekly + threshold q = 2 %"}


def report():
    """Summary table results/p3_summary.md and the four figures of docs/p3.md."""
    res = json.loads(RESULTS.read_text())
    lines = ["| Setting (BRF) | Seeds | Missed, mean | Missed, worst period | Released | Seeds with every period ≤ 1 % |",
             "|---|---|---|---|---|---|"]
    for s in SETTINGS:
        tb = table(s, res)
        if tb is None:
            continue
        m, r = tb
        lines.append(f"| {NAMES[s]} | {len(m)} | {m.mean():.2f} % | {m.mean(0).max():.2f} % | {r.mean():.1f} % | "
                     f"{int((m.max(1) <= 1).sum())} / {len(m)} |")
    text = ("# P3 results on the development data\n\nDays 1-64 only: train on days 1-22, evaluate on days 23-64 in "
            "5 periods (time order). Missed = released true defects / all true defects; released = released false "
            "calls / all false calls; mean over the 5 periods and the seeds; worst period = highest period mean. "
            "Made by `python scripts/p3_pipeline.py report` from `results/p3_results.json`.\n\n" + "\n".join(lines) + "\n")
    Path("results/p3_summary.md").write_text(text)
    print(text)
    figures(res)


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, GREY, LGREY, RED, GREEN = "#2F6DB5", "#555F6D", "#B8C4D6", "#C0392B", "#2E8B57"
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    out = Path("docs/img")

    # 1 true defects per production day, days 1-64
    d = pd.read_csv("data/dataset.csv", usecols=["class"])
    half = len(d) // 2
    g = pd.Series(d["class"].values[:half]).groupby(production_day(load_days()[:half])).sum()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(g.index, g.values, color=[LGREY if x <= TRAIN_DAYS else BLUE for x in g.index])
    ax.axvline(TRAIN_DAYS + 0.5, color=GREY, ls="--")
    ax.set_xlabel("Production day"); ax.set_ylabel("True defects per day")
    ax.set_title("Training days 1-22 (grey), evaluation days 23-64 (blue)", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "p3_days.png", dpi=150); plt.close(fig)

    # 2 missed per period
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, s, c in [("Method of [7]: trained once", "once", GREY), ("+ weekly retraining", "weekly", BLUE),
                        ("+ threshold q = 0", "q0", GREEN)]:
        ax.plot(range(1, 6), table(s, res)[0].mean(0), marker="o", color=c, lw=2.5, label=label)
    ax.axhline(1, color=RED, ls="--"); ax.set_xticks(range(1, 6), [f"P{i}" for i in range(1, 6)])
    ax.set_ylabel("Missed true defects (%)"); ax.set_xlabel("Evaluation period (days 23-64), mean of the seeds")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "p3_periods.png", dpi=150); plt.close(fig)

    # 3 retraining intervals
    sched = [(n, s) for n, s in [("Never", "once"), ("Per period", "period"), ("Weekly", "weekly"),
                                 ("3 days", "3days"), ("Daily", "daily")] if table(s, res) is not None]
    fig, ax = plt.subplots(figsize=(10, 4.5)); x = np.arange(len(sched))
    ax.bar(x - 0.2, [table(s, res)[0].mean() for _, s in sched], 0.4, color=BLUE, label="Missed, mean")
    ax.bar(x + 0.2, [table(s, res)[0].mean(0).max() for _, s in sched], 0.4, color=LGREY, label="Missed, worst period")
    ax.axhline(1, color=RED, ls="--")
    ax.set_xticks(x, [f"{n}\n({len(table(s, res)[0])} seeds)" for n, s in sched]); ax.set_ylabel("Missed true defects (%)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "p3_retrain.png", dpi=150); plt.close(fig)

    # 4 threshold trade-off
    qs = [(n, s) for n, s in [("q = 0", "q0"), ("0.1 %", "q0.1"), ("0.2 %", "q0.2"), ("0.5 %", "q0.5"),
                              ("1 %", "q1"), ("2 %", "q2")] if table(s, res) is not None]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    X = [table(s, res)[1].mean() for _, s in qs]
    ax.plot(X, [table(s, res)[0].mean() for _, s in qs], "o-", color=BLUE, lw=2.5, label="Missed, mean")
    ax.plot(X, [table(s, res)[0].mean(0).max() for _, s in qs], "s-", color=LGREY, lw=2, label="Missed, worst period")
    for (n, s), xx in zip(qs, X):
        ax.annotate(n, (xx, table(s, res)[0].mean(0).max()), textcoords="offset points", xytext=(4, 6), fontsize=11)
    w = table("weekly", res)
    ax.plot([w[1].mean()], [w[0].mean()], "D", color=GREY, label="CV threshold of [7], weekly")
    ax.axhline(1, color=RED, ls="--"); ax.set_xlabel("False calls released (%)"); ax.set_ylabel("Missed true defects (%)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "p3_threshold.png", dpi=150); plt.close(fig)


def check():
    """Re-run seed 1 of three settings and compare with the stored results (must be identical)."""
    X, y, days, h = data()
    p, seed = seed_params()[0]
    res = json.loads(RESULTS.read_text())
    for setting in ["once", "weekly", "q0"]:
        new = run(p, seed, X, y, days, h, setting)
        old = res[setting][str(seed)]
        same = all(abs(a["missed"] - b["missed"]) < 1e-12 and abs(a["released"] - b["released"]) < 1e-12
                   for a, b in zip(new, old))
        print(f"{setting:7s} missed per period {[round(r['missed'] * 100, 2) for r in new]} %  same as stored: {same}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"run": lambda: run_all(*sys.argv[2:]), "report": report, "check": check}.get(cmd, lambda: print(__doc__))()
