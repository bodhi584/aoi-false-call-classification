"""Figure 2 of the bibliography report: the groups of Table 1 as points.
Numbers from Pfab & Rothering (2025), Table III (p. 6), Table V (p. 11), Table IV (p. 7),
and our baseline (results/replication_pfab2025.json)."""
import matplotlib.pyplot as plt

G = "#1D6B3A"
fig, ax = plt.subplots(figsize=(7.4, 5.4))
# F: target zone of [7] and A: Siemens target (40 % line)
ax.add_patch(plt.Rectangle((0, 40), 1, 60, color="#DFF0E3", zorder=0))
ax.axvline(1, color=G, ls="--", lw=1.2, zorder=1)
ax.axhline(40, color=G, ls="--", lw=1.2, zorder=1)
ax.text(1.15, 2, "1 % missed", fontsize=8, color=G)
ax.text(15.9, 41.5, "40 % released", fontsize=8, color=G, ha="right")

# B: test set, ideal threshold (1 % missed by construction)
for name, y in [("XGBoost", 92.5), ("RF", 90.1), ("BRF", 79.6)]:
    ax.scatter(1, y, marker="D", s=40, facecolor="white", edgecolor="#14213D", zorder=4)
ax.scatter([], [], marker="D", s=40, facecolor="white", edgecolor="#14213D", label="B  test set, ideal threshold (XGBoost, RF, BRF)")
# C: test set, threshold as set by [7]
ax.scatter([0.8, 1.0], [79.7, 89.7], marker="*", s=170, color="#2F6DB5", edgecolor="#14213D", zorder=5,
           label="C  test set, threshold of [7] (BRF, RF)")
# D: five later periods of [7]
brf = ([7.3, 5.3, 8.6, 9.4, 9.3], [67.5, 48.7, 65.3, 78.0, 41.5])
rf = ([10.2, 14.6, 15.0, 10.9, 11.0], [66.7, 52.4, 65.4, 76.0, 39.7])
ax.scatter(*brf, s=36, color="#2F6DB5", zorder=3, label="D  5 later periods of [7], BRF")
ax.scatter(*rf, s=36, color="#9AB4D6", zorder=3, label="D  5 later periods of [7], RF")
# E: our baseline, mean of the later periods
ax.scatter([7.5, 9.1], [57.9, 55.8], marker="s", s=48, color="#C0392B", zorder=4,
           label="E  our baseline, later periods, mean (BRF, RF)")

ax.set_xlim(0, 16); ax.set_ylim(0, 100)
# inset: zoom on the test-set results B and C
ins = ax.inset_axes([0.17, 0.67, 0.25, 0.3])
ins.add_patch(plt.Rectangle((0, 40), 1, 60, color="#DFF0E3", zorder=0))
ins.axvline(1, color=G, ls="--", lw=1)
for name, y in [("XGBoost", 92.5), ("RF", 90.1), ("BRF", 79.6)]:
    ins.scatter(1, y, marker="D", s=30, facecolor="white", edgecolor="#14213D", zorder=4)
    ins.text(1.06, y, f"{name} {y}", fontsize=6.5, va="center")
for name, x, y in [("BRF", 0.8, 79.7), ("RF", 1.0, 89.7)]:
    ins.scatter(x, y, marker="*", s=110, color="#2F6DB5", edgecolor="#14213D", zorder=5)
    ins.text(x - 0.06, y - 2.2, f"{name} {y}", fontsize=6.5, ha="right", va="center")
ins.set_xlim(0.4, 1.75); ins.set_ylim(75, 96)
ins.set_title("test set, zoomed", fontsize=7); ins.tick_params(labelsize=6)
ax.indicate_inset_zoom(ins, edgecolor="#555F6D")
ax.set_xlabel("Missed true defects (%)"); ax.set_ylabel("False calls released = manual checks saved (%)")
ax.plot([], [], color=G, ls="--", lw=1.2, label="A, F  limits: 40 % released (A, F), 1 % missed (F)")
ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="#DFF0E3", label="F  target zone"))
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("../img/benefit_pfab2025.png", dpi=200)
