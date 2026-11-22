"""P3: which inputs matter? Filter method (Pearson correlation) vs. embedded method (SHAP on the trained model).

    python scripts/p3_feature_relevance.py  ->  results/p3_feature_relevance.json, docs/img/slides/corr_matrix.png, shap_share.png

1 correlation  Pearson correlation matrix of the 70 measurements and the class, training rows (days 1-22)
2 shap         SHAP values (TreeExplainer, Lundberg & Lee 2017) of the model of setting "once", seed 42,
               on the 20 % held-out rows of days 1-22; share of each input in the total mean |SHAP|
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent))
from p3_pipeline import MODEL, data, seed_params                      # noqa: E402
from replicate_pfab2025 import fit                                     # noqa: E402
from p3_model_example import input_names                               # noqa: E402

OUT = Path("results/p3_feature_relevance.json")


def main():
    X, y, days, h = data()
    df = pd.read_csv("data/dataset.csv", index_col=0, nrows=h)
    num = [c for c in df.columns if c.startswith("inspection_feat")]
    d = df[num + ["class"]].loc[:, df[num + ["class"]].std() > 0]
    C = d.corr(method="pearson")
    rc = C["class"].drop("class").abs()
    F = C.drop(index="class", columns="class").abs().values
    iu = np.triu_indices_from(F, 1)
    out = {"rows": int(h), "true_defects": int(y[:h].sum()),
           "correlation": {"measurements_with_variation": int(len(rc)), "constant_measurements": [c for c in num if c not in rc.index],
                           "max_abs_r_with_class": float(rc.max()), "argmax": rc.idxmax(),
                           "n_abs_r_with_class_below_0.1": int((rc < 0.1).sum()),
                           "pairs_between_measurements": int(len(iu[0])), "pairs_abs_r_above_0.8": int((F[iu] > 0.8).sum()),
                           "abs_r_with_class": rc.round(4).to_dict()}}
    p, seed = seed_params()[0]
    i_fit, i_val = train_test_split(np.arange(h), test_size=0.2, stratify=y[:h], random_state=seed)
    model = fit(MODEL, p, seed, X[i_fit], y[i_fit])
    sc, rf, names = model.named_steps["scaler"], model.named_steps["model"], input_names()
    sv = shap.TreeExplainer(rf).shap_values(sc.transform(X[i_val]), check_additivity=True)[:, :, 1]   # class 1 = true defect
    ms = np.abs(sv).mean(0); share = ms / ms.sum(); order = np.argsort(-share); cum = np.cumsum(share[order])
    out["shap"] = {"explained_rows": int(len(i_val)), "true_defects_in_them": int(y[i_val].sum()),
                   "share": {names[i]: float(share[i]) for i in order},
                   "inputs_for_50pct": int(np.searchsorted(cum, 0.5) + 1), "inputs_for_80pct": int(np.searchsorted(cum, 0.8) + 1),
                   "inputs_for_95pct": int(np.searchsorted(cum, 0.95) + 1), "inputs_with_zero_shap": int((ms == 0).sum()),
                   "spearman_shap_vs_gini_importance": float(spearmanr(ms, rf.feature_importances_)[0])}
    m70 = [names.index(c) for c in rc.index]
    out["shap"]["spearman_shap_vs_abs_r_70_measurements"] = float(spearmanr(ms[m70], rc.values)[0])
    keep = [n for n in rc.index if rc[n] >= 0.1]                        # what a correlation filter |r| >= 0.1 would keep
    out["shap"]["filter_abs_r_ge_0.1_keeps"] = keep
    out["shap"]["filter_abs_r_ge_0.1_shap_share_kept"] = float(sum(share[names.index(n)] for n in keep))
    out["shap"]["top10_abs_r_below_0.1"] = int(sum(1 for i in order[:10] if names[i] in rc.index and rc[names[i]] < 0.1))
    OUT.write_text(json.dumps(out, indent=1))
    figures(out, C)
    print(json.dumps({k: v for k, v in out["correlation"].items() if k != "abs_r_with_class"}, indent=1))
    print({k: v for k, v in out["shap"].items() if k != "share"})
    print([(n.replace("inspection_", ""), round(s * 100, 1)) for n, s in list(out["shap"]["share"].items())[:15]])


def figures(o, C):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch, Rectangle
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8A94A6", "xtick.color": "#3A4A63", "ytick.color": "#14213D", "axes.labelcolor": "#3A4A63"})
    BLUE, NAVY, RED, BG = "#2F6DB5", "#14213D", "#C0392B", "#FBFCFE"
    dst = Path("docs/img/slides"); dst.mkdir(parents=True, exist_ok=True)
    cols = ["class"] + [c for c in C.columns if c != "class"]
    M = C.loc[cols, cols]
    lab = ["class"] + [c.replace("inspection_feat", "") for c in cols[1:]]
    fig, ax = plt.subplots(figsize=(8.4, 7.4), facecolor=BG)
    sns.heatmap(M, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True, ax=ax, xticklabels=lab, yticklabels=lab,
                cbar_kws={"label": "Pearson r", "shrink": 0.8})
    step = [i for i in range(len(lab)) if i == 0 or i % 5 == 0]
    ax.set_xticks([i + 0.5 for i in step], [lab[i] for i in step], rotation=90, fontsize=11)
    ax.set_yticks([i + 0.5 for i in step], [lab[i] for i in step], rotation=0, fontsize=11)
    ax.add_patch(Rectangle((0, 0), len(lab), 1, fill=False, ec=NAVY, lw=2.2))
    ax.set_xlabel("measurement (inspection_feat number)"); ax.set_ylabel("")
    fig.savefig(dst / "corr_matrix.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
    sh, r = o["shap"]["share"], o["correlation"]["abs_r_with_class"]
    top = list(sh)[:12][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 6.6), facecolor=BG); ax.set_facecolor(BG)
    ax.barh([n.replace("inspection_", "") for n in top], [sh[n] * 100 for n in top],
            color=[NAVY if "=" in n else BLUE for n in top], height=0.66)
    for i, n in enumerate(top):
        rr = f"|r| = {r[n]:.3f}" if n in r else "category"
        ax.text(sh[n] * 100 + 0.15, i, f"{sh[n] * 100:.1f} %", va="center", fontsize=13, color=NAVY)
        ax.text(12.6, i, rr, va="center", ha="right", fontsize=13, color=RED if n in r and r[n] < 0.1 else "#3A4A63")
    ax.text(12.6, len(top) - 0.1, "correlation\nwith class", ha="right", va="bottom", fontsize=12, color="#3A4A63")
    ax.set_xlim(0, 12.8); ax.set_ylim(-0.6, len(top) + 0.9)
    ax.set_xlabel("share of the model's decisions (mean |SHAP|, %)")
    ax.legend(handles=[Patch(color=BLUE, label="measurement"), Patch(color=NAVY, label="category value (0/1)")],
              frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=12)
    fig.savefig(dst / "shap_share.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
