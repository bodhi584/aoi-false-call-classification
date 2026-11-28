"""Root-cause analysis (statistical, descriptive): which factors are associated with true defects?

The feature names are anonymised, so this finds associations, not physical causes.
All rows (440,274), and both halves separately to check that an association is stable.

    python scripts/p4_root_cause.py  ->  results/p4_root_cause.json, docs/img/rca/*.png

A  70 measurements vs class: Pearson r, Spearman rho, single-feature AUROC, mutual information,
   strength class by |r| (Cohen: < 0.1 negligible, 0.1-0.3 weak, 0.3-0.5 moderate, >= 0.5 strong),
   permutation importance of the model (P4) for comparison; also the strongest |r| within each inspection type.
B  5 categorical features vs class: Cramer's V, defect rate per level, levels with the highest lift.
C  Bursts: days with >= 60 true defects, share of the top component code and inspection type among their defects.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

OUT = Path("results/p4_root_cause.json")
CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]


def strength(r):
    a = abs(r)
    return "strong" if a >= 0.5 else "moderate" if a >= 0.3 else "weak" if a >= 0.1 else "negligible"


def cramers_v(x, y):
    tab = pd.crosstab(x, y)
    chi2 = chi2_contingency(tab, correction=False)[0]
    return float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))


def main():
    df = pd.read_csv("data/dataset.csv", index_col=0)
    n_dev = len(df) // 2
    df["half"] = np.where(np.arange(len(df)) < n_dev, "dev", "test")
    y = df["class"].to_numpy()
    num = [c for c in df.columns if c.startswith("inspection_feat") and df[c].std() > 0]
    fi = json.loads(Path("results/p4_feature_importance.json").read_text())["all"]["importance"]
    fi_rank = {k: i + 1 for i, k in enumerate(fi)}

    # ---- A: measurements
    rng = np.random.default_rng(0)
    sub = np.sort(np.concatenate([np.flatnonzero(y == 1), rng.choice(np.flatnonzero(y == 0), 60000, replace=False)]))
    mi = mutual_info_classif(df[num].to_numpy()[sub], y[sub], random_state=0)
    rows = []
    for j, c in enumerate(num):
        x = df[c].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1])
        rho = float(spearmanr(x, y)[0])
        auc = float(roc_auc_score(y, x))
        r_dev = float(np.corrcoef(x[:n_dev], y[:n_dev])[0, 1]); r_test = float(np.corrcoef(x[n_dev:], y[n_dev:])[0, 1])
        within = {}
        for t, g in df.groupby("inspection_type"):
            if g[c].std() > 0 and g["class"].sum() >= 20:
                within[str(t)] = float(np.corrcoef(g[c], g["class"])[0, 1])
        best_t = max(within, key=lambda k: abs(within[k])) if within else None
        rows.append({"feature": c, "pearson_r": r, "spearman_rho": rho, "auroc": auc,
                     "auroc_sep": abs(auc - 0.5) * 2, "mutual_info": float(mi[j]),
                     "r_dev": r_dev, "r_test": r_test, "strength": strength(r),
                     "within_type_best": best_t, "within_type_r": within.get(best_t) if best_t else None,
                     "within_type_strength": strength(within[best_t]) if best_t else None,
                     "perm_importance": fi.get(c), "perm_rank": fi_rank.get(c)})
    rows.sort(key=lambda d: -abs(d["pearson_r"]))
    counts = pd.Series([d["strength"] for d in rows]).value_counts().to_dict()
    counts_within = pd.Series([d["within_type_strength"] for d in rows if d["within_type_strength"]]).value_counts().to_dict()

    # ---- B: categoricals
    cats = []
    rate = y.mean()
    for c in CAT:
        g = df.groupby(c)["class"].agg(["sum", "count"])
        g["rate"] = g["sum"] / g["count"]
        g["lift"] = g["rate"] / rate
        top = g[g["count"] >= 500].sort_values("rate", ascending=False).head(5)
        cats.append({"feature": c, "levels": int(len(g)), "cramers_v": cramers_v(df[c], df["class"]),
                     "cramers_v_dev": cramers_v(df[c][:n_dev], df["class"][:n_dev]),
                     "cramers_v_test": cramers_v(df[c][n_dev:], df["class"][n_dev:]),
                     "top_levels": [{"level": str(k), "joints": int(v["count"]), "true_defects": int(v["sum"]),
                                     "rate_pct": round(100 * v["rate"], 2), "lift": round(v["lift"], 1)}
                                    for k, v in top.iterrows()],
                     "share_of_defects_top5_levels_pct": round(100 * g.sort_values("sum", ascending=False)["sum"].head(5).sum() / y.sum(), 1)})
    for d in cats:
        d["strength"] = strength(d["cramers_v"])
    cats.sort(key=lambda d: -d["cramers_v"])

    # ---- C: bursts
    ts = pd.to_datetime(df["timestamp"])
    df["day"] = ((ts.dt.normalize() - ts.dt.normalize().min()).dt.days).to_numpy()
    td = df[df["class"] == 1]
    per_day = td.groupby("day").size()
    bursts = []
    for d, n in per_day[per_day >= 60].items():
        t = td[td["day"] == d]
        mf = t["meta_feat1"].value_counts(); it = t["inspection_type"].value_counts()
        bursts.append({"day": int(d), "half": t["half"].iloc[0], "true_defects": int(n),
                       "top_component": str(mf.index[0]), "top_component_share_pct": round(100 * mf.iloc[0] / n, 1),
                       "top_inspection_type": str(it.index[0]), "top_type_share_pct": round(100 * it.iloc[0] / n, 1),
                       "components": int(t["meta_feat1"].nunique())})
    in_bursts = per_day[per_day >= 60].sum()

    out = {"rows": int(len(df)), "true_defects": int(y.sum()), "defect_rate_pct": round(100 * rate, 3),
           "measurements": rows, "strength_counts": counts, "strength_counts_within_type": counts_within,
           "categoricals": cats, "bursts": bursts,
           "burst_days": len(bursts), "defects_in_burst_days": int(in_bursts),
           "defects_in_burst_days_pct": round(100 * in_bursts / y.sum(), 1),
           "days_with_defects": int(len(per_day))}
    OUT.write_text(json.dumps(out, indent=1))
    figures(out)
    print({k: v for k, v in out.items() if k not in ("measurements", "categoricals", "bursts")})
    for d in rows[:10]:
        print(d["feature"], round(d["pearson_r"], 3), round(d["spearman_rho"], 3), round(d["auroc"], 3),
              round(d["mutual_info"], 4), d["strength"], d["within_type_best"], d["within_type_r"] and round(d["within_type_r"], 3), d["perm_rank"])
    for d in cats:
        print(d["feature"], d["levels"], round(d["cramers_v"], 3), d["strength"], d["top_levels"][:2])
    for b in bursts:
        print(b)


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, BLUE, GOLD, RED, GRAY, GREEN = "#14213D", "#2F6DB5", "#B8870A", "#BF382B", "#8A93A0", "#2E7D4F"
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    dst = Path("docs/img/rca"); dst.mkdir(parents=True, exist_ok=True)
    # 1 strength of every factor: |r| of measurements (overall and best within type) and Cramer's V of categoricals
    m = out["measurements"][:15]; c = out["categoricals"]
    names = [d["feature"] for d in c] + [d["feature"].replace("inspection_", "") for d in m]
    vals = [d["cramers_v"] for d in c] + [abs(d["pearson_r"]) for d in m]
    within = [None] * len(c) + [abs(d["within_type_r"]) if d["within_type_r"] is not None else None for d in m]
    yy = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(names) + 1.8))
    ax.barh(yy, vals, color=[GOLD] * len(c) + [BLUE] * len(m), height=0.62)
    for i, w in zip(yy, within):
        if w is not None:
            ax.plot(w, i, "D", color=NAVY, ms=6)
    for x0, lab in ((0.1, "weak"), (0.3, "moderate"), (0.5, "strong")):
        ax.axvline(x0, color=GRAY, ls=":", lw=1); ax.text(x0, len(names) - 0.3, f" {lab}", color=GRAY, fontsize=10)
    for i, v in zip(yy, vals):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    ax.set_yticks(yy, names)
    ax.set_xlim(0, max(0.55, max(vals) * 1.2))
    ax.set_xlabel("Association with true defect\ngold: Cramér's V (categorical) · blue: |Pearson r| (all rows) · ◆ best |r| within one inspection type")
    ax.set_title("Association strength: nothing moderate or strong;\nthe component code (meta_feat1) is the strongest factor")
    fig.tight_layout(); fig.savefig(dst / "association_strength.png", dpi=150); plt.close(fig)
    # 2 bursts: share of the top component among the defects of each burst day
    b = out["bursts"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(b))
    ax.bar(x, [d["top_component_share_pct"] for d in b], color=[BLUE if d["half"] == "dev" else GOLD for d in b], width=0.6)
    for i, d in enumerate(b):
        ax.text(i, d["top_component_share_pct"], f"{d['true_defects']}\nTD", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [f"day {d['day']}" for d in b], rotation=45)
    ax.set_ylabel("Share of the day's true defects\nfrom ONE component code (%)"); ax.set_ylim(0, 115)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="development half"), Patch(color=GOLD, label="test half")],
              frameon=False, loc="upper center", ncol=2)
    ax.set_ylim(0, 125)
    ax.set_title(f"Bursts: {out['burst_days']} days hold {out['defects_in_burst_days_pct']} % of all true defects;\n"
                 "in 10 of 14 one component code gives 40 % or more of the day's defects")
    fig.tight_layout(); fig.savefig(dst / "bursts_by_component.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--figures"]:                   # redraw from the saved result only
        figures(json.loads(OUT.read_text()))
    else:
        main()
