"""Figure 1 of the literature report: the reviewed literature from general to specific.

Usage: python docs/literature/literature_map.py   (writes docs/img/literature_map.png)
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

LEVELS = [
    ("Zero-defect manufacturing", "detect, repair, predict, prevent defects", "[1]"),
    ("AOI in electronics assembly", "camera + fixed limits check every solder joint", "[2], [3]"),
    ("False calls of AOI", "most flagged boards are good; manual re-inspection", "[4], [5]"),
    ("Second check with supervised machine learning", "classifiers on AOI measurements", "[6]–[10]"),
    ("Evaluation in time order", "imbalanced classes, recall, specificity, drift", "[8], [11]–[15]"),
]
COLORS = ["#E9EDF3", "#DCE4F0", "#CBD9EC", "#B7CBE6", "#2F6DB5"]

fig, ax = plt.subplots(figsize=(10, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
h = 0.95
for i, ((head, body, refs), c) in enumerate(zip(LEVELS, COLORS)):
    top, bot = 5.4 - i * h, 5.4 - (i + 1) * h + 0.08
    wt, wb = 6.6 - i * 0.9, 6.6 - (i + 1) * 0.9 + 0.12
    ax.add_patch(Polygon([(3.6 - wt / 2, top), (3.6 + wt / 2, top), (3.6 + wb / 2, bot), (3.6 - wb / 2, bot)],
                         closed=True, fc=c, ec="none"))
    tc = "white" if i == len(LEVELS) - 1 else "#14213D"
    ax.text(3.6, (top + bot) / 2 + 0.13, head, ha="center", va="center", fontsize=10, fontweight="bold", color=tc)
    ax.text(3.6, (top + bot) / 2 - 0.2, body, ha="center", va="center", fontsize=8, color=tc)
    ax.text(7.35, (top + bot) / 2, refs, ha="left", va="center", fontsize=10, color="#1F3A5F", fontweight="bold")
ax.annotate("", xy=(8.9, 0.6), xytext=(8.9, 5.3), arrowprops=dict(arrowstyle="-|>", color="#555F6D", lw=1.5))
ax.text(9.1, 5.2, "general", fontsize=9, color="#555F6D", va="top")
ax.text(9.1, 0.7, "specific\n(our\nproject)", fontsize=9, color="#2F6DB5", va="bottom", fontweight="bold")
ax.text(0.0, 0.05, "Gap addressed by our project: evaluation in time order with a fixed minimum recall (99 %) for true defects.",
        fontsize=9, color="#A12A2A")
fig.tight_layout()
fig.savefig("docs/img/literature_map.png", dpi=200)
