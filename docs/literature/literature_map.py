"""Figures of the literature report.

Figure 1: the reviewed literature from general to specific.
Figure 2: missed true defects reported by Pfab & Rothering (2025), Tables IV and V (values copied from the paper).

Usage: python docs/literature/literature_map.py   (writes docs/img/literature_map.png and docs/img/pfab2025_slip.png)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

LEVELS = [
    ("Zero-defect manufacturing", "detect, repair, predict, prevent defects", "[1]"),
    ("AOI in electronics assembly", "camera + fixed limits check every solder joint", "[2], [3]"),
    ("False calls of AOI", "most flagged boards are good; manual re-inspection", "[4], [5]"),
    ("Second check with supervised machine learning", "works on a test set, fails on later days", "[6]–[8]"),
    ("Classifiers for tabular data", "which families exist and how they compare", "[9]–[18]"),
    ("Evaluation with rare true defects", "recall, specificity, threshold", "[19]–[21]"),
    ("Why models fail over time: drift", "theory, manufacturing cases", "[22]–[28]"),
    ("Remedy: retraining", "time-order tests, related cases", "[29]–[31]"),
]
COLORS = ["#EEF1F6", "#E5EAF2", "#DAE2EE", "#CED9EA", "#C1D0E5", "#B3C6E0", "#9DB6D8", "#2F6DB5"]


def literature_map():
    fig, ax = plt.subplots(figsize=(10, 5.8))
    n = len(LEVELS)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.0); ax.axis("off")
    h = 0.8
    for i, ((head, body, refs), c) in enumerate(zip(LEVELS, COLORS)):
        top, bot = 6.8 - i * h, 6.8 - (i + 1) * h + 0.06
        wt, wb = 7.2 - i * 0.56, 7.2 - (i + 1) * 0.56 + 0.08
        ax.add_patch(Polygon([(3.6 - wt / 2, top), (3.6 + wt / 2, top), (3.6 + wb / 2, bot), (3.6 - wb / 2, bot)],
                             closed=True, fc=c, ec="none"))
        tc = "white" if i == n - 1 else "#14213D"
        ax.text(3.6, (top + bot) / 2 + 0.11, head, ha="center", va="center", fontsize=9, fontweight="bold", color=tc)
        ax.text(3.6, (top + bot) / 2 - 0.17, body, ha="center", va="center", fontsize=7.3, color=tc)
        ax.text(7.45, (top + bot) / 2, refs, ha="left", va="center", fontsize=9.5, color="#1F3A5F", fontweight="bold")
    ax.annotate("", xy=(8.9, 0.5), xytext=(8.9, 6.7), arrowprops=dict(arrowstyle="-|>", color="#555F6D", lw=1.5))
    ax.text(9.1, 6.6, "general", fontsize=9, color="#555F6D", va="top")
    ax.text(9.1, 0.55, "specific\n(our\nproject)", fontsize=9, color="#2F6DB5", va="bottom", fontweight="bold")
    ax.text(0.0, 0.05, "Our question: does periodic retraining keep the second check working on later production days?",
            fontsize=9, color="#A12A2A")
    fig.tight_layout()
    fig.savefig("docs/img/literature_map.png", dpi=200)


def pfab_figure():
    labels = ["Test set\n(same period)", "Slice 1", "Slice 2", "Slice 3", "Slice 4", "Slice 5"]
    brf, brf_sd = [0.8, 7.3, 5.3, 8.6, 9.4, 9.3], [0.6, 2.5, 1.6, 3.1, 1.0, 1.7]
    rf, rf_sd = [1.0, 10.2, 14.6, 15.0, 10.9, 11.0], [1.2, 7.4, 2.5, 5.6, 3.3, 4.6]
    x, w = np.arange(6), 0.38
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    b1 = ax.bar(x - w / 2, brf, w, yerr=brf_sd, capsize=3, color="#9AA5B1", label="Balanced random forest")
    b2 = ax.bar(x + w / 2, rf, w, yerr=rf_sd, capsize=3, color="#2F6DB5", label="Random forest")
    ax.bar_label(b1, [f"{v:.1f}" for v in brf], fontsize=7, padding=2)
    ax.bar_label(b2, [f"{v:.1f}" for v in rf], fontsize=7, padding=2)
    ax.axhline(1, color="#C0392B", ls="--", lw=1)
    ax.text(5.6, 1.4, "target ≤ 1 %", color="#C0392B", fontsize=8, ha="right")
    ax.axvline(0.5, color="#555F6D", lw=0.8, ls=":")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Missed true defects (%)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig("docs/img/pfab2025_slip.png", dpi=200)


if __name__ == "__main__":
    literature_map()
    pfab_figure()
