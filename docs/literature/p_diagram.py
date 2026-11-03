"""P-diagram (parameter diagram) of our system for the bibliography report."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
DARK, MID, LIGHT = "#14213D", "#2F6DB5", "#DCE7F5"

def box(x, y, w, h, title, body, fc, ec=MID):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9, fontweight="bold", color=DARK)
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=7.6, color=DARK, linespacing=1.35)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555F6D", lw=1.4))

box(0.1, 2.45, 2.2, 2.1, "Input", "Siemens AOI data [4]\nflagged joints, time order\n70 measurements\n+ 5 categories\nlabel: true defect\nor false call", "#FBFCFE")
box(3.3, 2.25, 3.3, 2.5, "System", "supervised binary classifier\n(random forest, balanced RF)\n\n+ decision threshold selection\n+ concept drift adaptation\n   (retraining)", LIGHT)
box(7.6, 2.45, 2.3, 2.1, "Output (responses)", "per later period k:\nmissed true defects\nFNR$_k$ ≤ 1 %\nfalse calls released\nTNR$_k$ ≥ 40 %", "#E3F2E8", "#1D6B3A")
arrow(2.3, 3.5, 3.3, 3.5); arrow(6.6, 3.5, 7.6, 3.5)

box(1.6, 5.25, 6.8, 1.6, "Control factors (varied in our test plan)",
    "A  threshold method (3 levels)     B  retraining schedule (6 levels)\n"
    "C  training window (2 levels)       D  model (2 levels)\n"
    "held constant as in [7]: data split, features, hyper-parameters, recall target 99 %", "#FBFCFE")
box(1.6, 0.1, 6.8, 1.6, "Noise factors (not controlled)",
    "drift: change of the data and of the share of true defects (0.09–3.11 % per period)\n"
    "label errors from the manual station [4]      mix of inspection types\n"
    "random seed (replicated with the 10 seeds of [7])", "#FBE9E7", "#C0392B")
arrow(4.95, 5.25, 4.95, 4.75); arrow(4.95, 1.7, 4.95, 2.25)
fig.tight_layout(); fig.savefig("../img/p_diagram.png", dpi=200)
