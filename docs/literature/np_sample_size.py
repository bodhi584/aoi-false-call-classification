"""Figure for the bibliography report: how many true defects a left-out sample needs so that a
threshold keeps the miss rate below alpha with confidence 1 - delta (Tong et al. 2018):
n >= log(delta) / log(1 - alpha)."""
import math
import matplotlib.pyplot as plt

alphas = [0.005, 0.01, 0.02, 0.05, 0.10]
plt.figure(figsize=(7, 3.6))
for delta, c in [(0.05, "#2F6DB5"), (0.10, "#9AB4D6")]:
    n = [math.ceil(math.log(delta) / math.log(1 - a)) for a in alphas]
    bars = plt.bar([i + (0.2 if delta == 0.10 else -0.2) for i in range(len(alphas))], n, 0.4,
                   color=c, label=f"confidence {1 - delta:.0%}")
    for b, v in zip(bars, n):
        plt.text(b.get_x() + b.get_width() / 2, v + 8, str(v), ha="center", fontsize=8)
plt.xticks(range(len(alphas)), [f"{a:.1%}".replace(".0%", "%") for a in alphas])
plt.xlabel("Allowed share of missed true defects (alpha)")
plt.ylabel("True defects needed")
plt.title("Minimum true defects in the left-out data (Tong et al. 2018)", fontsize=10)
plt.legend(frameon=False)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("../img/np_sample_size.png", dpi=200)
