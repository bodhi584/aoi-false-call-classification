"""P3: do single measurements relate linearly to the class? Pearson correlation on the whole data set (filter method [23]).

    python scripts/p3_feature_relevance.py  ->  results/p3_feature_relevance.json, docs/img/slides/corr_matrix.png

Pearson correlation matrix of the measurements that vary and the class, all 440,274 rows.
"""
import json
from pathlib import Path

import pandas as pd

OUT = Path("results/p3_feature_relevance.json")


def main():
    da = pd.read_csv("data/dataset.csv", index_col=0, usecols=lambda c: c == "Unnamed: 0" or c.startswith("inspection_feat") or c == "class")
    num = [c for c in da.columns if c.startswith("inspection_feat")]
    da = da[num + ["class"]].loc[:, da[num + ["class"]].std() > 0]
    C = da.corr(method="pearson")
    r = C["class"].drop("class").abs()
    out = {"correlation_all_rows": {"rows": int(len(da)), "true_defects": int(da["class"].sum()), "measurements_with_variation": int(len(r)),
                                    "max_abs_r_with_class": float(r.max()), "argmax": r.idxmax(),
                                    "n_abs_r_with_class_below_0.1": int((r < 0.1).sum()), "abs_r_with_class": r.round(4).to_dict()}}
    OUT.write_text(json.dumps(out, indent=1))
    figure(C)
    print({k: v for k, v in out["correlation_all_rows"].items() if k != "abs_r_with_class"})


def figure(C):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8A94A6", "xtick.color": "#3A4A63", "ytick.color": "#14213D", "axes.labelcolor": "#3A4A63"})
    NAVY, BG = "#14213D", "#FBFCFE"
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


if __name__ == "__main__":
    main()
