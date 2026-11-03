"""Figure for the bibliography report: what the second check of Pfab & Rothering (2025) reaches.
Numbers from their Table V (test set, p. 11) and Table IV (five later periods, p. 7)."""
import matplotlib.pyplot as plt

later = {"Balanced RF": ([7.3, 5.3, 8.6, 9.4, 9.3], [67.5, 48.7, 65.3, 78.0, 41.5]),
         "RF": ([10.2, 14.6, 15.0, 10.9, 11.0], [66.7, 52.4, 65.4, 76.0, 39.7])}
test = {"Balanced RF": (0.8, 79.7), "RF": (1.0, 89.7)}
col = {"Balanced RF": "#2F6DB5", "RF": "#9AB4D6"}

fig, ax = plt.subplots(figsize=(7, 4))
ax.add_patch(plt.Rectangle((0, 40), 1, 60, color="#DFF0E3", zorder=0))
ax.axvline(1, color="#1D6B3A", ls="--", lw=1.2, zorder=1)
ax.axhline(40, color="#1D6B3A", ls="--", lw=1.2, zorder=1)
ax.text(1.15, 2, "1 % missed", fontsize=8, color="#1D6B3A")
ax.text(15.9, 41.5, "40 % released", fontsize=8, color="#1D6B3A", ha="right")
ax.annotate("target of [7]: ≤ 1 % missed, ≥ 40 % released", xy=(0.5, 70), xytext=(2.2, 22),
            fontsize=8, color="#1D6B3A", arrowprops=dict(arrowstyle="->", color="#1D6B3A"))
for m in later:
    ax.scatter(*later[m], color=col[m], s=40, label=f"{m}, 5 later periods", zorder=3)
    ax.scatter(*test[m], color=col[m], marker="*", s=180, edgecolor="#14213D", label=f"{m}, test set", zorder=4)
ax.set_xlim(0, 16); ax.set_ylim(0, 100)
ax.set_xlabel("Missed true defects (%)"); ax.set_ylabel("False calls released = manual checks saved (%)")
ax.set_title("Second check of [7]: on target on the test set, not on later periods", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("../img/benefit_pfab2025.png", dpi=200)
