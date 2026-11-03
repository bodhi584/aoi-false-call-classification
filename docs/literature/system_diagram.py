"""Figure for the bibliography report: our system = classifier + two adaptation strategies."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.set_xlim(0, 12); ax.set_ylim(-0.25, 7.2); ax.axis("off")
DARK, MID, LIGHT, GREEN, RED, ORANGE = "#14213D", "#2F6DB5", "#DCE7F5", "#1D6B3A", "#C0392B", "#B9770E"

def box(x, y, w, h, text, fc="#FBFCFE", ec=MID, fs=8, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=fc, ec=ec, lw=1.2))
    lines = text.split("\n")
    ax.text(x + w / 2, y + h - 0.2, lines[0], ha="center", va="top", fontsize=fs + 0.5, fontweight="bold", color=DARK)
    if len(lines) > 1:
        ax.text(x + w / 2, y + h - 0.55, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs, color=DARK, linespacing=1.3)

def arrow(x1, y1, x2, y2, c="#555F6D", ls="-", txt=None, tx=0, ty=0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3, ls=ls))
    if txt:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, txt, fontsize=7.2, color=c, ha="center")

# main flow (middle row)
box(0.1, 3.1, 2.0, 1.4, "Flagged joints\nperiod k\nAOI measurements x", fs=7.5)
box(2.8, 3.1, 2.2, 1.4, "Classifier f$_k$\nrandom forest or\nbalanced random forest", fc=LIGHT, fs=7.5)
box(5.7, 3.1, 2.1, 1.4, "Decision rule\nscore s(x) ≥ t$_k$ ?", fc=LIGHT, fs=7.5)
box(8.6, 4.3, 1.8, 0.95, "yes: manual station\n(person checks)", fs=7.2)
box(8.6, 2.35, 1.8, 0.95, "no: released\n(no manual check)", fs=7.2)
box(10.7, 3.0, 1.25, 1.6, "Output\nFNR$_k$ ≤ 1 %\nTNR$_k$ ≥ 40 %", fc="#E3F2E8", ec=GREEN, fs=7.2)
arrow(2.1, 3.8, 2.8, 3.8); arrow(5.0, 3.8, 5.7, 3.8, txt="s(x)", ty=0.12)
arrow(7.8, 4.0, 8.6, 4.7); arrow(7.8, 3.6, 8.6, 2.85)
arrow(10.4, 4.7, 10.7, 4.1); arrow(10.4, 2.85, 10.7, 3.4)

# strategy 1 (top)
box(4.4, 5.55, 4.8, 1.5, "Strategy 1: decision threshold selection\nsets the threshold t$_k$ on labelled data\nbaseline: once, mean of 5 CV thresholds at 99 % recall [7]\nNeyman–Pearson threshold [14] · prior shift adjustment per period [15]",
    fc="#FDF2E3", ec=ORANGE, fs=7.2)
arrow(6.75, 5.55, 6.75, 4.5, c=ORANGE, txt="t$_k$", tx=0.18)

# strategy 2 (bottom)
box(1.2, 0.15, 5.0, 1.55, "Strategy 2: concept drift adaptation\nretrains the classifier f$_k$ on labelled data\nbaseline: train once [7]\nperiodic retraining [12], [17], [18] · drift-triggered retraining [16]",
    fc="#FDF2E3", ec=ORANGE, fs=7.2)
arrow(3.9, 1.7, 3.9, 3.1, c=ORANGE, txt="new f$_k$", tx=0.38)

# labels feed back
box(7.0, 0.35, 3.3, 1.15, "Labelled joints of earlier periods\ntraining window: all data so far\nor a sliding window [12]", fs=7.2)
arrow(9.2, 2.35, 9.2, 1.5, c="#555F6D", ls="--", txt="labels*", tx=0.32)
arrow(7.0, 0.92, 6.2, 0.92)
ax.text(0.1, -0.05, "* labels from the manual station; released joints get no manual label, so the simulation assumes that their labels arrive later (Section 2.5)", fontsize=6.8, color="#555F6D")
fig.tight_layout(); fig.savefig("../img/system_diagram.png", dpi=200)
