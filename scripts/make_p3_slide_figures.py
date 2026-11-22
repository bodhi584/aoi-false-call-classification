"""P3: every chart of the P3 slides, made only from the stored result files and the data.

    python scripts/make_p3_slide_figures.py   ->  docs/img/slides/*.png

Inputs:   results/p3_results.json        all settings x 10 seeds x 5 periods (from p3_pipeline.py run)
          results/p3_model_example.json  tree 1 of seed 42 (from p3_model_example.py)
          data/dataset.csv               true defects per production day
          Table IV of [7]                slip and volume reduction of the BRF of [7] (numbers typed in below)
Charts:   target7       objective: target region of [7] and the 5 slices of [7]
          dev_days      input: true defects per production day, days 1-64
          dev_periods   output: missed per period, [7] vs. + feature 1 vs. + feature 2
          dev_retrain   output: retraining schedules
          dev_q         output: threshold q vs. the CV threshold of [7]
          raw_missed    output: every run, missed per seed and period
          gini          system: Gini of the first split of tree 1
          model_choice  design: three steps from the data fact to the random forest (diagram, no data)
          brf_principle design: how the balanced random forest samples and votes (diagram, no data)
          drift_evidence design: share of true defects and change of the inputs per week (results/p3_drift.json)
          np_formula    design: the threshold rule of [14, Alg. 1] as a typeset formula
          np_threshold  design: threshold as a quantile of the true-defect scores (schematic, no data)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

NAVY, BLUE, GREY, LGREY, RED, GREEN, BG = "#14213D", "#2F6DB5", "#555F6D", "#B8C4D6", "#C0392B", "#2E8B57", "#FBFCFE"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 17, "axes.edgecolor": GREY, "axes.labelcolor": NAVY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
OUT = Path("docs/img/slides")
RES = json.load(open("results/p3_results.json"))
SEEDS = list(RES["once"])                                   # the 10 seeds of [7], in the order of data/runs.csv


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


def res(setting):
    """Arrays seeds x 5 periods, in %: missed true defects and released false calls."""
    runs = [RES[setting][s] for s in SEEDS]
    return (np.array([[p["missed"] for p in r] for r in runs]) * 100,
            np.array([[p["released"] for p in r] for r in runs]) * 100)


def target7():
    # [7], Sect. VI, Table IV, model BRFC2: slip rate and volume reduction per evaluation slice (mean of 10 runs)
    slip = [7.3, 5.3, 8.6, 9.4, 9.3]; vol = [67.5, 48.7, 65.3, 78.0, 41.5]
    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    ax.add_patch(Rectangle((40, 0), 60, 1, color=GREEN, alpha=0.22, lw=0))
    ax.text(70, 0.45, "target of [7]: ≤ 1 % missed, ≥ 40 % released", color=GREEN, fontsize=14, ha="center", va="center", fontweight="bold")
    ax.axhline(1, color=RED, ls="--", lw=2); ax.axvline(40, color=GREY, ls=":", lw=1.6)
    ax.text(1.5, 1.25, "1 % limit (hard)", color=RED, fontsize=14)
    ax.text(39.2, 2.4, "40 % (reference)", color=GREY, fontsize=14, ha="right")
    ax.scatter(vol, slip, s=150, color=NAVY, zorder=3)
    for i, (x, y) in enumerate(zip(vol, slip), 1):
        ax.text(x + 1.4, y + 0.15, f"slice {i}", fontsize=14, color=NAVY)
    ax.text(70, 11.3, "all 5 slices: 5.3–9.4 % missed", fontsize=16, color=NAVY, fontweight="bold", ha="center")
    ax.set_xlim(0, 100); ax.set_ylim(0, 12)
    ax.set_xlabel("False calls released (%)"); ax.set_ylabel("True defects missed (%)")
    save(fig, "target7.png")


def dev_days():
    d = pd.read_csv("data/dataset.csv", usecols=["timestamp", "class"])
    day = pd.to_datetime(d.timestamp).dt.date
    di = day.map({x: i + 1 for i, x in enumerate(sorted(day.unique()))}).values   # production day 1, 2, ...
    h = len(d) // 2                                                                 # days 1-64
    g = pd.DataFrame({"d": di[:h], "y": d["class"].values[:h]}).groupby("d").y.agg(["sum", "count"])
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    col = [RED if v >= 200 else (LGREY if x <= 22 else BLUE) for x, v in zip(g.index, g["sum"])]
    ax.bar(g.index, g["sum"], color=col, width=0.8)
    ax.axvline(22.5, color=GREY, ls="--", lw=1.5)
    ax.text(11, 290, "Train\ndays 1–22", ha="center", color=GREY, fontsize=16)
    ax.text(50, 290, "Evaluation slices, days 23–64\n5 slices in time order", ha="center", color=BLUE, fontsize=16)
    for x in g.index[g["sum"] >= 200]:
        ax.text(x + {24: -0.3, 29: -1.6, 31: 1.8}[x], g["sum"][x] + 6, f"day {x}: {g['sum'][x]}",
                ha={24: "center", 29: "right", 31: "left"}[x], color=RED, fontsize=13)
    ax.set_ylim(0, 340); ax.set_xlim(0, 64.5)
    ax.set_xlabel("Production day"); ax.set_ylabel("True defects per day")
    save(fig, "dev_days.png")


def dev_periods():
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    p = np.arange(1, 6)
    for name, key, c, lw in [("[7] method: trained once", "once", GREY, 3), ("+ weekly retraining", "weekly", BLUE, 3.5),
                             ("+ new threshold (q = 1 %)", "q1", GREEN, 3.5)]:
        v = res(key)[0].mean(0)
        ax.plot(p, v, color=c, lw=lw, marker="o", ms=8)
        for xx, yy in zip(p, v):
            ax.text(xx, yy + {GREY: 0.55, BLUE: 0.55, GREEN: -1.25}[c], f"{yy:.1f}", color=c, ha="center", fontsize=12, fontweight="bold")
        ax.text(5.12, {BLUE: 3.4, GREEN: 1.9}.get(c, v[-1]), name, color=c, va="center", fontsize=16, fontweight="bold")
    ax.axhline(1, color=RED, ls="--", lw=2); ax.text(6.6, 0.8, "1 % limit", color=RED, fontsize=15, va="top")
    ax.set_xticks(p, [f"P{i}" for i in p]); ax.set_xlim(0.8, 7.6); ax.set_ylim(-1.8, 19)
    ax.set_yticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5]); ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("Missed true defects (%)"); ax.set_xlabel("Evaluation period (days 23–64, time order), mean of 10 seeds")
    save(fig, "dev_periods.png")


def dev_retrain():
    keys = [("Never [7]", "once"), ("Per period", "period"), ("Weekly", "weekly"), ("Every 3 days", "3days"), ("Daily", "daily")]
    fig, ax = plt.subplots(figsize=(11.5, 5.2)); x = np.arange(len(keys))
    mean = [res(k)[0].mean() for _, k in keys]; worst = [res(k)[0].mean(0).max() for _, k in keys]; rel = [res(k)[1].mean() for _, k in keys]
    ax.bar(x - 0.2, mean, 0.4, color=BLUE, label="Mean of 5 periods"); ax.bar(x + 0.2, worst, 0.4, color=LGREY, label="Worst period")
    for i in x:
        ax.text(i - 0.2, mean[i] + 0.3, f"{mean[i]:.1f}", ha="center", color=BLUE, fontsize=15, fontweight="bold")
        ax.text(i + 0.2, worst[i] + 0.3, f"{worst[i]:.1f}", ha="center", color=GREY, fontsize=15)
        ax.text(i, -3.6, f"released {rel[i]:.0f} %", ha="center", color=NAVY, fontsize=14)
    ax.axhline(1, color=RED, ls="--", lw=2); ax.text(4.45, 1.5, "1 %", color=RED, fontsize=15)
    ax.set_xticks(x, [k for k, _ in keys]); ax.set_ylabel("Missed true defects (%)"); ax.set_ylim(0, 20); ax.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(bottom=0.2)
    save(fig, "dev_retrain.png")


def dev_q():
    pts = [("CV, 99 % recall ([7])", "weekly"), ("q = 2 %", "q2"), ("q = 1 %", "q1"), ("q = 0.5 %", "q0.5"),
           ("q = 0.2 %", "q0.2"), ("q = 0.1 %", "q0.1"), ("q = 0", "q0")]
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    X = [res(k)[1].mean() for _, k in pts]; M = [res(k)[0].mean() for _, k in pts]; W = [res(k)[0].mean(0).max() for _, k in pts]
    ax.plot(X[1:], W[1:], color=LGREY, lw=2.5, marker="s", ms=9, label="Worst period")
    ax.plot(X[1:], M[1:], color=BLUE, lw=3, marker="o", ms=9, label="Mean of 5 periods")
    ax.scatter([X[0]] * 2, [M[0], W[0]], color=[BLUE, LGREY], marker="D", s=110, zorder=3)
    ax.text(X[0] + 0.7, W[0], pts[0][0], color=GREY, fontsize=14, va="center")
    for (n, _), xx, ww in list(zip(pts, X, W))[1:]:
        if n == "q = 0":
            ax.text(xx - 0.5, ww - 0.15, n, ha="right", color=NAVY, fontsize=13)
        else:
            ax.text(xx + 0.5, ww - 0.45, n, ha="left", color=NAVY, fontsize=13)
    ax.axhline(1, color=RED, ls="--", lw=2); ax.text(36.5, 1.25, "1 % limit", color=RED, fontsize=15)
    i1 = [n for n, _ in pts].index("q = 1 %")                      # same nominal level: CV 99 % recall vs q = 1 %
    ax.scatter([X[i1]], [M[i1]], s=260, facecolors="none", edgecolors=NAVY, linewidths=2.2, zorder=4)
    ax.scatter([X[0]], [M[0]], s=260, facecolors="none", edgecolors=NAVY, linewidths=2.2, zorder=4)
    ax.annotate("", xy=(X[i1] + 0.4, M[i1] + 0.05), xytext=(X[0] - 0.4, M[0] - 0.02), arrowprops=dict(arrowstyle="->", color=NAVY, lw=2))
    ax.text(26.5, 0.15, f"same 99 % recall level:\n{M[0]:.1f} % → {M[i1]:.1f} % missed", color=NAVY, fontsize=13, ha="center", fontweight="bold")
    ax.set_xlim(0, 42); ax.set_ylim(0, 9); ax.set_xlabel("False calls released (%), mean of 10 seeds"); ax.set_ylabel("Missed true defects (%)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "dev_q.png")


def raw_missed():
    """Every single run: missed true defects per seed (rows) and period (columns), red above 1 %."""
    sets = [("[7] method: trained once", "once"), ("+ (1) weekly retraining", "weekly"), ("+ (2) threshold q = 1 %", "q1")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), sharey=True)
    for ax, (title, key) in zip(axes, sets):
        m, r = res(key)
        ax.imshow(np.where(m > 1, 1, 0), cmap=matplotlib.colors.ListedColormap(["#E6EEF7", "#F4D6D2"]), aspect="auto", vmin=0, vmax=1)
        for i in range(m.shape[0]):
            for j in range(5):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=13,
                        color=RED if m[i, j] > 1 else NAVY, fontweight="bold" if m[i, j] > 1 else "normal")
        ax.set_xticks(range(5), [f"P{j}" for j in range(1, 6)]); ax.tick_params(length=0)
        ax.set_yticks(range(len(SEEDS)), SEEDS)
        ax.set_title(f"{title}\nmean {m.mean():.1f} % missed, {r.mean():.0f} % released\n{(m > 1).sum()} of 50 runs × periods over 1 %", fontsize=14, color=NAVY)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Seed of [7] (hyper-parameter set)")
    save(fig, "raw_missed.png")


def gini():
    ex = json.load(open("results/p3_model_example.json"))
    t = ex["tree1"]; L, R = str(t["0"]["yes"]), str(t["0"]["no"])
    cnt = lambda k: (round(t[k]["true_defect_share"] * t[k]["weighted"]), round((1 - t[k]["true_defect_share"]) * t[k]["weighted"]), t[k]["gini"])
    cut, name = t["0"]["cut"], t["0"]["input"].replace("inspection_", "")
    rows = [("Parent", *cnt("0")), (f"Left child\n({name} ≤ {cut:.3f})", *cnt(L)), (f"Right child\n({name} > {cut:.3f})", *cnt(R))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.9), gridspec_kw={"width_ratios": [1, 1.15]})
    p = np.linspace(0, 1, 201); a1.plot(p * 100, 1 - p**2 - (1 - p)**2, color=NAVY, lw=2.2)
    for (lab, td, fc, g), c, txt, (dx, dy, ha) in zip(rows, [GREY, BLUE, RED], ["parent", "left", "right"],
                                                      [(0, 0.03, "center"), (-3, -0.12, "right"), (3, -0.02, "left")]):
        pp = td / (td + fc)
        a1.scatter([pp * 100], [g], s=70, color=c, zorder=3)
        a1.text(pp * 100 + dx, g + dy, f"{txt} {pp * 100:.0f} %, G = {g:.3f}" if txt == "parent" else f"{txt} {pp * 100:.0f} %\nG = {g:.3f}",
                fontsize=13, color=c, ha=ha)
    a1.set_xlim(0, 100); a1.set_ylim(0, 0.62); a1.set_xlabel("p = share of true defects in the group (%)"); a1.set_ylabel("Gini impurity G")
    a1.text(25, 0.6, "max 0.5 = half / half", ha="center", fontsize=13, color=GREY); a1.text(3, 0.03, "0 = pure", fontsize=13, color=GREY)
    a1.set_title("G = 1 − p² − (1 − p)²", fontsize=17, color=NAVY, loc="left")
    y = [2, 1, 0]
    for yi, (lab, td, fc, g) in zip(y, rows):
        n = td + fc
        a2.barh(yi, td / n, color=RED, height=0.55); a2.barh(yi, fc / n, left=td / n, color=BLUE, height=0.55)
        a2.text(1.03, yi, f"n = {n:,}\nG = {g:.3f}", va="center", fontsize=13, color=NAVY)
        a2.text(0.02, yi, f"{td} TD", va="center", fontsize=12, color="white", fontweight="bold")
        a2.text(td / n + 0.02, yi, f"{fc} FC", va="center", fontsize=12, color="white", fontweight="bold")
    a2.set_yticks(y); a2.set_yticklabels([r[0] for r in rows], fontsize=13); a2.set_xlim(0, 1.32)
    a2.set_xticks([0, 0.5, 1]); a2.set_xticklabels(["0", "50 %", "100 %"])
    a2.set_title(f"First split of tree 1, seed {ex['seed']}", fontsize=17, color=NAVY, loc="left")
    a2.text(0, -0.65, "TD = true defects, FC = false calls (counted with repeats)", fontsize=13, color=GREY, va="top")
    a2.set_ylim(-1.0, 2.5)
    save(fig, "gini.png")


def model_choice():
    """Diagram, no data: the three choices that lead from the data fact to the random forest."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 9); ax.set_ylim(0, 4.6); ax.axis("off")
    rows = [(3.95, "1", "Linear or\nnon-linear?", [None, ("linear model", "no single measurement is\nlinearly related: |r| ≤ 0.18", 0), ("non-linear model", "", 1)]),
            (2.45, "2", "Which non-linear\nmodel?", [None, ("neural network", "not robust to\nuninformative features [8]", 0), ("tree-based model", "", 1)]),
            (0.95, "3", "Which tree-based\nmodel?", [("single tree", "unstable,\noverfits [L5]", 0), ("boosting", "less robust to\nlabel noise [10]", 0), ("random forest", "", 1)])]
    w, h, xs = 2.05, 0.62, [2.35, 4.6, 6.85]
    for y, n, q, opts in rows:
        ax.text(0.05, y, n, fontsize=22, color=BLUE, fontweight="bold", va="center")
        ax.text(0.42, y, q, fontsize=15.5, color=NAVY, fontweight="bold", va="center", linespacing=1.15)
        for x, o in zip(xs, opts):
            if o is None:
                continue
            name, why, ok = o
            ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", lw=1.8,
                                        fc=BLUE if ok else BG, ec=BLUE if ok else LGREY))
            ax.text(x + w / 2, y, name, ha="center", va="center", fontsize=16, color="white" if ok else GREY,
                    fontweight="bold" if ok else "normal")
            if not ok:
                ax.text(x + w / 2, y - h / 2 - 0.10, why, ha="center", va="top", fontsize=12.5, color=GREY, linespacing=1.15)
        if y > 1:
            ax.annotate("", xy=(xs[2] + w / 2, y - 1.5 + h / 2 + 0.08), xytext=(xs[2] + w / 2, y - h / 2 - 0.08),
                        arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 2.2, "mutation_scale": 18})
    fig.subplots_adjust(0, 0, 1, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "model_choice.png", dpi=200); plt.close(fig)


def brf_principle():
    """Diagram, no data: balanced samples per tree and the vote of the forest [10, Sect. 2.2]."""
    from matplotlib.patches import Circle, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 9); ax.set_ylim(0, 4.6); ax.axis("off")
    arrow = {"arrowstyle": "-|>", "color": GREY, "lw": 1.6, "mutation_scale": 14}
    # 1 training data: a long bar of false calls, a thin slice of true defects
    ax.text(0.95, 4.32, "Training data", ha="center", fontsize=15, color=NAVY, fontweight="bold")
    ax.add_patch(Rectangle((0.45, 0.35), 1.0, 3.1, fc=BLUE, ec="none"))
    ax.add_patch(Rectangle((0.45, 3.49), 1.0, 0.14, fc=RED, ec="none"))
    ax.text(0.95, 1.9, "false calls: about 99 %", ha="center", va="center", fontsize=13.5, color="white", rotation=90)
    ax.text(0.95, 3.86, "true defects: about 1 %", ha="center", va="center", fontsize=12, color=NAVY)
    # 2 one balanced sample per tree
    ax.text(3.55, 4.32, "One sample per tree", ha="center", fontsize=15, color=NAVY, fontweight="bold")
    ys = [3.45, 2.45, 0.95]; names = ["1", "2", "T"]
    for y, n in zip(ys, names):
        ax.add_patch(Rectangle((2.75, y - 0.2), 0.8, 0.4, fc=RED, ec="none"))
        ax.add_patch(Rectangle((3.57, y - 0.2), 0.8, 0.4, fc=BLUE, ec="none"))
        ax.text(3.15, y, "n", ha="center", va="center", fontsize=14, color="white", fontweight="bold")
        ax.text(3.97, y, "n", ha="center", va="center", fontsize=14, color="white", fontweight="bold")
        ax.annotate("", xy=(2.68, y), xytext=(1.55, 2.3 + (y - 2.3) * 0.35), arrowprops=arrow)
        # 3 the tree grown on this sample
        x0 = 5.55
        for dx, dy in [(-0.32, -0.3), (0.32, -0.3)]:
            ax.plot([x0, x0 + dx], [y + 0.18, y + 0.18 + dy], color=GREY, lw=1.6, zorder=1)
            ax.add_patch(Circle((x0 + dx, y + 0.18 + dy), 0.09, fc=BG, ec=GREY, lw=1.6, zorder=2))
        ax.add_patch(Circle((x0, y + 0.18), 0.09, fc=BG, ec=GREY, lw=1.6, zorder=2))
        ax.text(x0, y - 0.38, f"tree {n}", ha="center", va="center", fontsize=12.5, color=GREY)
        ax.annotate("", xy=(5.05, y), xytext=(4.45, y), arrowprops=arrow)
        ax.annotate("", xy=(6.95, 2.3 + (y - 2.3) * 0.25), xytext=(6.1, y), arrowprops=arrow)
    ax.plot([3.55] * 3, [1.55, 1.72, 1.89], ls="none", marker="o", ms=3.5, color=GREY)
    ax.plot([5.55] * 3, [1.55, 1.72, 1.89], ls="none", marker="o", ms=3.5, color=GREY)
    ax.text(3.55, 0.30, "n true defects (bootstrap)\n+ n false calls (random draw)", ha="center", va="center", fontsize=12.5,
            color=NAVY, linespacing=1.15)
    ax.text(5.55, 4.32, "T trees", ha="center", fontsize=15, color=NAVY, fontweight="bold")
    # 4 the vote
    ax.text(7.95, 4.32, "Vote", ha="center", fontsize=15, color=NAVY, fontweight="bold")
    ax.add_patch(FancyBboxPatch((7.05, 1.55), 1.8, 1.5, boxstyle="round,pad=0.02,rounding_size=0.12", lw=1.8, fc=BLUE, ec=BLUE))
    ax.text(7.95, 2.3, "score s(x) =\nshare of trees\nvoting\n“true defect”", ha="center", va="center", fontsize=13.5, color="white",
            fontweight="bold", linespacing=1.2)
    fig.subplots_adjust(0, 0, 1, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "brf_principle.png", dpi=200); plt.close(fig)


def drift_evidence():
    """Data fact for the drift slide: p(y) and p(X) both change from week to week (whole data set).

    Top: share of true defects. Bottom: share of the joints of inspection types 2 and 3.
    """
    d = json.load(open("results/p3_drift.json"))
    w = np.array(d["weeks"]); share = np.array(d["share_pct"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.32})
    a1.bar(w, share, color=BLUE, width=0.72)
    mean = 100 * sum(d["true_defects"]) / sum(d["rows"])
    a1.axhline(mean, color=GREY, lw=1.2, ls=(0, (4, 3)))
    a1.text(0.45, mean + 0.25, f"all weeks: {mean:.2f} %", fontsize=12, color=GREY, va="bottom")
    for i in (int(share.argmin()), int(share.argmax())):
        low = share[i] < mean
        a1.text(w[i] + (0 if low else 0.36), mean + 0.2 if low else share[i] + 0.15, f"{share[i]:.2f} %",
                ha="center" if low else "right", va="bottom", fontsize=12, color=NAVY)
    a1.set_ylim(0, share.max() * 1.22); a1.set_yticks([0, 2, 4, 6])
    a1.set_ylabel("True defects\n(% of joints)", fontsize=12.5)
    a1.set_title("The class share changes: p(y)", fontsize=14, color=NAVY, loc="left", fontweight="bold")
    ORANGE = "#C96A12"
    for k, col, dy in (("2", BLUE, 6), ("3", ORANGE, 6)):
        v = np.array(d["inspection_type_share_pct"][k])
        a2.plot(w, v, color=col, lw=2, marker="o", ms=5.5)
        i = int(v.argmax())
        a2.text(w[i], v[i] + dy, f"inspection type {k}", ha="center", va="bottom", fontsize=12, color=NAVY)
    a2.set_ylim(0, 92); a2.set_yticks([0, 25, 50, 75])
    a2.set_ylabel("Share of the\njoints (%)", fontsize=12.5)
    a2.set_title("The inputs change: p(X)", fontsize=14, color=NAVY, loc="left", fontweight="bold")
    a2.set_xticks(w); a2.set_xlim(0.4, 19.6); a2.set_xlabel("Production week", fontsize=12.5)
    for a in (a1, a2):
        a.tick_params(labelsize=12)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.93, bottom=0.13)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "drift_evidence.png", dpi=200); plt.close(fig)


def np_threshold():
    """Schematic, no data: the threshold t is an order statistic of the scores of left-out true defects [14, Alg. 1]."""
    from scipy.stats import beta
    td, fc = beta(6, 2.2), beta(1.6, 6)                      # illustrative shapes only
    rng = np.random.default_rng(3)
    left_out = np.sort(td.rvs(60, random_state=rng))         # scores of left-out true defects (illustration)
    t = left_out[0]
    x = np.linspace(0, 1, 600)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(x, fc.pdf(x), where=x <= t, color=BLUE, alpha=0.22, lw=0)
    ax.fill_between(x, td.pdf(x), where=x <= t, color=RED, alpha=0.85, lw=0)
    ax.plot(x, fc.pdf(x), color=BLUE, lw=2.5)
    ax.plot(x, td.pdf(x), color=RED, lw=2.5)
    top = 5.0
    ax.plot(left_out, np.full(60, -0.22), ls="none", marker="|", ms=11, mew=1.4, color=RED, clip_on=False)
    ax.plot([t], [-0.22], ls="none", marker="|", ms=17, mew=3, color=NAVY, clip_on=False)
    ax.plot([t, t], [0, top * 0.88], color=NAVY, lw=2, ls=(0, (5, 3)))
    ax.text(t, top * 0.90, "threshold t", ha="center", va="bottom", fontsize=16, color=NAVY, fontweight="bold")
    ax.text(t - 0.02, top * 0.78, "s(x) < t: release", ha="right", fontsize=14, color=GREY)
    ax.text(t + 0.02, top * 0.78, "s(x) ≥ t: operator checks", ha="left", fontsize=14, color=GREY)
    ax.text(0.045, 1.0, "false calls", ha="left", fontsize=15, color=NAVY)
    ax.text(0.80, 1.0, "true defects", ha="center", fontsize=15, color=NAVY)
    ax.annotate("released true defects:\nat most α, with probability ≥ 1 − δ", xy=(t - 0.02, 0.03), xytext=(0.47, 3.3), fontsize=14,
                color=NAVY, ha="left", va="center", arrowprops={"arrowstyle": "-|>", "color": NAVY, "lw": 1.5})
    ax.text(0.70, -0.66, "ticks: scores of left-out true defects; t is the k-th lowest", ha="center", va="center", fontsize=12.5, color=NAVY)
    ax.set_xlim(0, 1); ax.set_ylim(0, top)
    ax.set_yticks([]); ax.set_ylabel("Number of joints", fontsize=15)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["0", "1"]); ax.tick_params(axis="x", pad=14)
    ax.text(0.5, -1.25, "Score s(x) = share of trees voting “true defect”", ha="center", va="center", fontsize=14, color=NAVY)
    ax.text(1.0, top, "Schematic, not data", ha="right", va="top", fontsize=13, color=GREY, style="italic")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.97, bottom=0.26)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "np_threshold.png", dpi=200); plt.close(fig)


def np_formula():
    """The threshold rule of [14, Alg. 1], typeset: the k-th lowest score of m left-out true defects."""
    tex = (r"$t = s_{(k)}, \qquad k = \max\left\{\, r \;:\; \sum_{j=0}^{r-1} \binom{m}{j}\, \alpha^{j} (1-\alpha)^{m-j} \;\leq\; \delta \right\}$")
    with plt.rc_context({"mathtext.fontset": "cm"}):
        fig = plt.figure(figsize=(8.7, 1.15))
        fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=23, color=NAVY)
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "np_formula.png", dpi=200, transparent=True); plt.close(fig)


if __name__ == "__main__":
    for f in [target7, dev_days, dev_periods, dev_retrain, dev_q, raw_missed, gini, model_choice, brf_principle, drift_evidence, np_threshold, np_formula]:
        f()
        print("docs/img/slides/" + f.__name__ + ".png")
