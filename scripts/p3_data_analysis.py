"""P3: understand the training data (days 1-22) and see which inputs the forest uses.

    python scripts/p3_data_analysis.py   ->  results/p3_data_analysis.json, docs/img/slides/train_*.png, forest_importance.png

1 categories    share of true defects per inspection_type and per value of meta_feat1
2 measurements  how well each of the 70 measurements separates true defects from false calls:
                correlation with the class, AUC, mutual information (all rows), and AUC inside one inspection type
3 rules         two-question rules found in the data (inspection type + one measurement)
4 forest        importance of the 206 inputs in the model of setting "once", seed 42 (same model as p3_model_example.py),
                inputs never used in any split, and inputs without any variation in the fit rows
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent))
from p3_pipeline import MODEL, data, seed_params                      # noqa: E402
from replicate_pfab2025 import CATEGORICAL, fit                        # noqa: E402
from p3_model_example import input_names                               # noqa: E402

OUT = Path("results/p3_data_analysis.json")
sep = lambda y, x: float(max(roc_auc_score(y, x), 1 - roc_auc_score(y, x)))   # direction-free AUC: 0.5 none, 1 perfect


def main():
    X, y, days, h = data()
    df = pd.read_csv("data/dataset.csv", index_col=0, nrows=h)        # training part: days 1-22
    assert (df["class"].values == y[:h]).all()
    out = {"rows": int(h), "true_defects": int(y[:h].sum()), "true_defect_share": float(y[:h].mean())}
    # 1 categories
    cat = {}
    for c in ["inspection_type"] + CATEGORICAL[:4]:
        g = df.groupby(c)["class"].agg(["size", "sum"])
        cat[c] = {"values_in_training": int(len(g)), "values_with_true_defects": int((g["sum"] > 0).sum()),
                  "rows_in_values_without_true_defects": int(g.loc[g["sum"] == 0, "size"].sum()),
                  "per_value": [[int(v), int(r["size"]), int(r["sum"])] for v, r in g.iterrows()]}
    out["categories"] = cat
    # 2 measurements
    num = [c for c in df.columns if c.startswith("inspection_feat")]
    corr = {c: (abs(float(np.corrcoef(df[c], df["class"])[0, 1])) if df[c].std() > 0 else 0.0) for c in num}
    auc = {c: sep(df["class"], df[c]) for c in num}
    cats = pd.get_dummies(df[CATEGORICAL].astype(str), prefix=CATEGORICAL)
    Xmi = pd.concat([df[num], cats], axis=1)
    mi = mutual_info_classif(Xmi.values, df["class"].values,
                             discrete_features=[False] * len(num) + [True] * cats.shape[1], random_state=0)
    out["measurements"] = {"abs_correlation": corr, "auc": auc,
                           "mutual_information": {k: float(v) for k, v in zip(Xmi.columns, mi)},
                           "auc_within_type": {}}
    for f, t in [("inspection_feat96", 2), ("inspection_feat95", 3), ("inspection_feat12", 3), ("inspection_feat48", 1)]:
        d = df[df.inspection_type == t]
        out["measurements"]["auc_within_type"][f] = {"type": t, "all_rows": auc[f], "within_type": sep(d["class"], d[f])}
    # 3 rules: inspection type + one measurement
    rules = []
    for t, f, op, cut in [(2, "inspection_feat96", ">=", 0.3), (3, "inspection_feat95", "<", 0.3)]:
        d = df[df.inspection_type == t]
        m = d[f] >= cut if op == ">=" else d[f] < cut
        rules.append({"type": t, "input": f, "rule": f"{op} {cut}", "rows_yes": int(m.sum()), "true_defects_yes": int(d.loc[m, "class"].sum()),
                      "rows_no": int((~m).sum()), "true_defects_no": int(d.loc[~m, "class"].sum())})
    out["rules"] = rules
    # 4 forest: model of setting "once", seed 42
    p, seed = seed_params()[0]
    idx_fit, _ = train_test_split(np.arange(h), test_size=0.2, stratify=y[:h], random_state=seed)
    model = fit(MODEL, p, seed, X[idx_fit], y[idx_fit])
    rf, names = model.named_steps["model"], input_names()
    imp = rf.feature_importances_
    used = np.zeros(len(names), int)
    for e in rf.estimators_:
        f = e.tree_.feature
        used += np.bincount(f[f >= 0], minlength=len(names))
    const = X[idx_fit].max(0) == X[idx_fit].min(0)
    order = np.argsort(-imp)
    groups = {}
    for n, v in zip(names, imp):
        g = n.split("=")[0] if "=" in n else "inspection_feat"
        groups[g] = groups.get(g, 0.0) + float(v)
    out["forest"] = {"importance": {n: float(v) for n, v in zip(names, imp)}, "top15": [names[i] for i in order[:15]],
                     "top20_share": float(imp[order[:20]].sum()), "groups": groups,
                     "inputs_never_used": int((used == 0).sum()), "inputs_without_variation_in_fit_rows": int(const.sum()),
                     "never_used_but_varying": [names[i] for i in np.flatnonzero((used == 0) & ~const)]}
    OUT.write_text(json.dumps(out, indent=1))
    figures(out)


def figures(o):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8A94A6", "xtick.color": "#3A4A63", "ytick.color": "#14213D", "axes.labelcolor": "#3A4A63"})
    BLUE, NAVY, GREY, RED, BG = "#2F6DB5", "#14213D", "#8A94A6", "#C0392B", "#FBFCFE"
    dst = Path("docs/img/slides"); dst.mkdir(parents=True, exist_ok=True)
    base = o["true_defect_share"] * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.0), facecolor=BG, gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.55})
    it = sorted(o["categories"]["inspection_type"]["per_value"])[::-1]
    for i, (v, n, k) in enumerate(it):
        a1.barh(i, k / n * 100, color=BLUE, height=0.62)
        a1.text(k / n * 100 + 0.02, i, f"{k / n * 100:.2f} %  ({k} of {n:,})", va="center", fontsize=13, color=NAVY)
    a1.set_yticks(range(len(it)), [f"type {v}" for v, _, _ in it])
    a1.axvline(base, color=RED, ls="--", lw=1.6); a1.text(base + 0.02, -0.62, f"all rows: {base:.2f} %", color=RED, fontsize=13, va="center")
    a1.set_xlim(0, 1.45); a1.set_ylim(-0.95, len(it) - 0.5); a1.set_facecolor(BG); a1.set_xlabel("% true defects")
    a1.set_title("inspection_type: share of true defects", fontsize=15, color=NAVY, loc="left", pad=12)
    m1 = sorted([r for r in o["categories"]["meta_feat1"]["per_value"] if r[0] in (73, 3, 15, 27, 28, 22)], key=lambda r: r[2] / r[1])
    for i, (v, n, k) in enumerate(m1):
        a2.barh(i, k / n * 100, color=BLUE, height=0.62)
        a2.text(k / n * 100 + 0.8, i, f"{k / n * 100:.1f} %  ({k} of {n:,})", va="center", fontsize=13, color=NAVY)
    a2.set_yticks(range(len(m1)), [f"= {v}" for v, _, _ in m1])
    a2.axvline(base, color=RED, ls="--", lw=1.6); a2.text(base + 1, -0.62, f"all rows: {base:.2f} %", color=RED, fontsize=13, va="center")
    a2.set_xlim(0, 75); a2.set_ylim(-0.95, len(m1) - 0.5); a2.set_facecolor(BG); a2.set_xlabel("% true defects")
    a2.set_title("meta_feat1: values with the most true defects", fontsize=15, color=NAVY, loc="left", pad=12)
    fig.savefig(dst / "train_categories.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
    w = o["measurements"]["auc_within_type"]; rows = list(w)[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=BG); ax.set_facecolor(BG)
    for i, f in enumerate(rows):
        a, b = w[f]["all_rows"], w[f]["within_type"]
        ax.plot([a, b], [i, i], color="#B8C4D6", lw=4, zorder=1, solid_capstyle="round")
        ax.scatter([a], [i], s=150, color=GREY, zorder=2, edgecolor=BG, lw=2); ax.scatter([b], [i], s=150, color=BLUE, zorder=3, edgecolor=BG, lw=2)
        ax.text(a - 0.012, i + 0.28, f"{a:.2f}", ha="center", fontsize=13, color="#3A4A63")
        ax.text(b + 0.012, i + 0.28, f"{b:.2f}", ha="center", fontsize=13, color=BLUE, fontweight="bold")
    ax.set_yticks(range(len(rows)), [f"{f.replace('inspection_', '')}  (type {w[f]['type']})" for f in rows])
    ax.set_xlim(0.48, 0.95); ax.set_ylim(-0.6, len(rows) - 0.2); ax.axvline(0.5, color="#B8C4D6", lw=1)
    ax.set_xlabel("separation of true defects and false calls (AUC; 0.5 = none, 1 = perfect)")
    ax.scatter([], [], s=110, color=GREY, label="all rows"); ax.scatter([], [], s=110, color=BLUE, label="inside its inspection type")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=13)
    fig.savefig(dst / "train_measurements.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
    imp = o["forest"]["importance"]; top = o["forest"]["top15"][::-1]
    fig, ax = plt.subplots(figsize=(8.6, 6.2), facecolor=BG); ax.set_facecolor(BG)
    ax.barh([n.replace("inspection_", "") for n in top], [imp[n] * 100 for n in top], color=[NAVY if "=" in n else BLUE for n in top], height=0.65)
    for i, n in enumerate(top):
        ax.text(imp[n] * 100 + 0.1, i, f"{imp[n] * 100:.1f} %", va="center", fontsize=13, color=NAVY)
    ax.set_xlim(0, 10); ax.set_xlabel("importance: share of all Gini decrease in the 108 trees (%)")
    ax.legend(handles=[Patch(color=BLUE, label="measurement (inspection_feat)"), Patch(color=NAVY, label="category value (0/1 input)")],
              frameon=False, loc="lower right", fontsize=13)
    fig.savefig(dst / "forest_importance.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
