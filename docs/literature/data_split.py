"""Figure for the bibliography report: how the Siemens data are split in time (as in Pfab & Rothering 2025).
Numbers computed from dataset.csv of Pfab et al. (2024): joints in time order, first half / five later periods."""
import matplotlib.pyplot as plt

parts = [("First half\n(training + test set)", 0, 84, 220137, 0.79, "#2F6DB5"),
         ("P1", 84, 97, 44027, 0.35, "#9AB4D6"), ("P2", 97, 104, 44027, 0.09, "#9AB4D6"),
         ("P3", 104, 112, 44028, 0.81, "#9AB4D6"), ("P4", 112, 124, 44027, 2.17, "#9AB4D6"),
         ("P5", 124, 132, 44028, 3.11, "#9AB4D6")]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
for name, d0, d1, n, share, c in parts:
    a1.barh(0, d1 - d0, left=d0, color=c, edgecolor="white", height=0.6)
    a1.text((d0 + d1) / 2, 0, f"{name}\n{n:,} joints" if d0 == 0 else name, ha="center", va="center", fontsize=8,
            color="white" if c == "#2F6DB5" else "#14213D")
    a2.bar((d0 + d1) / 2, share, width=(d1 - d0) * 0.9, color=c)
    a2.text((d0 + d1) / 2, share + 0.08, f"{share:.2f} %", ha="center", fontsize=7.5)
a1.set_yticks([]); a1.set_title("440,274 flagged joints in time order: first half (80 % training, 20 % test, random and stratified) and five later periods P1–P5\n(about 44,000 joints each)", fontsize=8.5)
a2.set_ylabel("True defects (% of joints)"); a2.set_xlabel("Production day")
a2.set_ylim(0, 3.6)
for a in (a1, a2): a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("../img/data_split.png", dpi=200)
