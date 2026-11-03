"""Figure for the bibliography report: technical route from the problem to the methods we test."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
DARK, MID, LIGHT, RED = "#14213D", "#2F6DB5", "#DCE7F5", "#C0392B"

def box(x, y, w, h, text, fc=LIGHT, ec=MID, tc=DARK, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", wrap=True)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555F6D", lw=1))

ax.text(1.0, 5.75, "Problem", ha="center", fontsize=9, fontweight="bold")
ax.text(3.45, 5.75, "Cause", ha="center", fontsize=9, fontweight="bold")
ax.text(6.25, 5.75, "Method (source)", ha="center", fontsize=9, fontweight="bold")
ax.text(8.95, 5.75, "Factor in our test plan", ha="center", fontsize=9, fontweight="bold")

box(0.05, 2.2, 1.9, 1.6, "Later periods:\n5–15 % true defects\nmissed instead of\n≤ 1 %\n(Table 1, D, E)", fc="#FBE9E7", ec=RED, fs=8)

box(2.45, 3.9, 2.0, 1.1, "Threshold set once\ncannot hold 1 %\n[14]")
box(2.45, 1.0, 2.0, 1.1, "Drift: the data\nchange over time\n[7], [12], [13]")
arrow(1.95, 3.3, 2.45, 4.4); arrow(1.95, 2.7, 2.45, 1.6)

meth = [(4.95, "Threshold with a safety margin [14]"), (4.25, "Threshold re-set per period [15]"),
        (2.15, "Retraining on a fixed schedule\n[12], [17], [18]"), (1.45, "Retraining when drift is detected [16]"),
        (0.75, "Retraining data: all data so far\nor a recent window [12], [18]")]
for y, txt in meth:
    box(4.95, y - 0.3, 2.6, 0.6, txt, fs=7.3)
for y in (4.95, 4.25):
    arrow(4.45, 4.45, 4.95, y)
for y in (2.15, 1.45, 0.75):
    arrow(4.45, 1.55, 4.95, y)

box(8.05, 3.85, 1.85, 1.35, "A  Threshold method\n3 levels:\nas in [7] corrected,\nsafety margin, re-set", fc="#FBFCFE", fs=7.5)
box(8.05, 1.55, 1.85, 1.2, "B  Retraining schedule\n6 levels:\nnever … daily, on drift", fc="#FBFCFE", fs=7.5)
box(8.05, 0.4, 1.85, 0.8, "C  Retraining data\n2 levels", fc="#FBFCFE", fs=7.5)
for y in (4.95, 4.25):
    arrow(7.55, y, 8.05, 4.5)
for y in (2.15, 1.45):
    arrow(7.55, y, 8.05, 2.15)
arrow(7.55, 0.75, 8.05, 0.8)
ax.text(5.0, 0.05, "Held constant as in [7]: data and split, features, hyper-parameters, recall target 99 %. Factor D: model (RF, BRF). "
        "Responses: missed true defects and false calls released per later period.", ha="center", fontsize=7, color="#555F6D")
fig.tight_layout(); fig.savefig("../img/method_route.png", dpi=200)
