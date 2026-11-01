"""Figures for the P2 presentation, made from the result files only.

Usage: python scripts/make_p2_figures.py
Inputs: results/data_facts.json, results/replication_pfab2025.json
Output: docs/img/data_drift.png, docs/img/replication_pfab2025.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NAVY, BLUE, GREY, RED, ORANGE = "#14213D", "#2F6DB5", "#9AA5B1", "#C0392B", "#E0A030"


def data_drift():
    """Flagged boards and share of defective boards per week."""
    w = json.loads(Path("results/data_facts.json").read_text())["weekly_boards_and_defective_boards"]
    weeks = sorted(w, key=int)
    boards = np.array([w[k][0] for k in weeks])
    share = np.array([100 * w[k][1] / w[k][0] for k in weeks])
    x = np.arange(1, len(weeks) + 1)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    b = ax.bar(x, share, color=[RED if s > 3 else BLUE for s in share])
    ax.bar_label(b, [f"{s:.1f}" for s in share], fontsize=7, padding=2)
    ax.set_ylabel("Defective boards (% of flagged boards)", fontsize=9)
    ax.set_xlabel("Production week", fontsize=9)
    ax.set_xticks(x)
    ax2 = ax.twinx()
    ax2.plot(x, boards, color=GREY, marker="o", ms=3, lw=1.2)
    ax2.set_ylabel("Flagged boards per week", fontsize=9, color="#555F6D")
    for a in (ax, ax2):
        a.spines[["top"]].set_visible(False)
    ax.set_title("The line changes over time: share of true defects per week (bars), flagged boards (line)", fontsize=10)
    fig.tight_layout()
    fig.savefig("docs/img/data_drift.png", dpi=200)


def replication():
    """One-to-one replication of Pfab & Rothering (2025): missed defects and false calls released per slice."""
    r = json.loads(Path("results/replication_pfab2025.json").read_text())
    arms = [("paper", "Paper, as logged (train once)", GREY, "o"),
            ("paper_exact", "(a) Paper setup, our rerun", "#555F6D", "s"),
            ("train_once", "(b) Train once, threshold re-estimated", ORANGE, "^"),
            ("retrain", "(c) Retrain before each slice", BLUE, "D")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    x = np.arange(1, 6)
    for j, (m, name) in enumerate([("RFC", "Random forest"), ("BRFC", "Balanced random forest")]):
        v = r[m]
        for i, (k, label, c, mk) in enumerate(arms):
            if k == "paper":
                slip = np.array([z["paper"]["slices_slip"] for z in v])
                vr = np.array([z["paper"]["slices_volume_reduction"] for z in v])
            else:
                slip = np.array([[s["slip"] for s in z[k]] for z in v])
                vr = np.array([[s["volume_reduction"] for s in z[k]] for z in v])
            off = (i - 1.5) * 0.07
            axs[0, j].errorbar(x + off, 100 * slip.mean(0), yerr=100 * slip.std(0, ddof=1), color=c, marker=mk,
                               capsize=2, lw=1.6, label=label)
            axs[1, j].errorbar(x + off, 100 * vr.mean(0), yerr=100 * vr.std(0, ddof=1), color=c, marker=mk,
                               capsize=2, lw=1.6)
        axs[0, j].axhline(1, color=RED, ls="--", lw=1)
        axs[1, j].axhline(40, color=RED, ls="--", lw=1)
        axs[0, j].set_title(f"{name}, 10 seeds", fontsize=10)
        axs[0, j].set_ylim(0, 20)
        axs[1, j].set_ylim(0, 100)
        axs[1, j].set_xticks(x, [f"Slice {i}" for i in x])
    axs[0, 0].set_ylabel("Missed true defects (%)\ntarget ≤ 1 %", fontsize=9)
    axs[1, 0].set_ylabel("False calls released (%)\ntarget ≥ 40 %", fontsize=9)
    for a in axs.flat:
        a.spines[["top", "right"]].set_visible(False)
    axs[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("One-to-one replication of Pfab & Rothering (2025), joint level, mean ± SD", fontsize=10)
    fig.tight_layout()
    fig.savefig("docs/img/replication_pfab2025.png", dpi=200)


if __name__ == "__main__":
    data_drift()
    replication()
