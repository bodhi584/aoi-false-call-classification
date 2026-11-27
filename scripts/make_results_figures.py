"""Figures for the P4 model selection, the P5 final test and the P6 improvement attempts.

Inputs:  results/p4_dev_forest.json, p4_dev_system2.json, p5_final_test.json,
         p6_dev_ideas.json, p6_dev_ideas2.json, p6_dev_spc.json, p6_final_test.json
Outputs: docs/img/p4/forest_tuning.png, docs/img/p4/system_m_audit.png,
         docs/img/p5/final_test_windows.png, docs/img/p6/dev_ideas.png, docs/img/p6/second_evaluation.png
Missed = released true defects / true defects; released = released false calls / false calls.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NAVY, BLUE, GOLD, RED, GRAY, GREEN = "#14213D", "#2F6DB5", "#B8870A", "#BF382B", "#8A93A0", "#2E7D4F"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    return json.loads(Path("results", name).read_text())


def released_total(days_or_windows):
    return 100 * sum(d["released"] for d in days_or_windows) / sum(d["fc"] for d in days_or_windows)


def by_setting(runs, key, recs="windows"):
    """Max missed over all periods and seeds, mean released total over seeds, per setting."""
    g = defaultdict(list)
    for r in runs:
        g[key(r)].append((max(p["missed_pct"] for p in r["periods"]), released_total(r[recs])))
    return {k: (max(a for a, _ in v), np.mean([b for _, b in v])) for k, v in g.items()}


def limit(ax, xmax):
    ax.axvline(1, color=RED, ls="--", lw=1.5)
    ax.text(1.03, ax.get_ylim()[1] * 0.97, "1 % limit", color=RED, va="top")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Worst period: true defects missed (%)")
    ax.set_ylabel("False calls released (%)")


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def forest_tuning():
    res = by_setting(load("p4_dev_forest.json")["runs"],
                     lambda r: tuple(sorted(r["setting"]["forest"].items())))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for k, (mm, rel) in res.items():
        f = dict(k)
        best = f == {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "sampling_strategy": "all"}
        default = f == {"n_estimators": 100, "max_features": "sqrt", "min_samples_leaf": 1, "sampling_strategy": "all"}
        ax.scatter(mm, rel, s=110 if best or default else 45, zorder=3,
                   color=GREEN if best else NAVY if default else (BLUE if mm <= 1 else GRAY))
        if best:
            ax.annotate("chosen: 300 trees, sqrt, leaf 5, 1:1", (mm, rel), xytext=(8, 4), textcoords="offset points", color=GREEN)
        if default:
            ax.annotate("library defaults", (mm, rel), xytext=(8, -12), textcoords="offset points", color=NAVY)
    ax.set_ylim(0, max(r for _, r in res.values()) * 1.15)
    limit(ax, max(m for m, _ in res.values()) * 1.05)
    ax.set_title("Forest tuning (F1–F4), 24 settings × 5 seeds, development half")
    save(fig, "docs/img/p4/forest_tuning.png")


def system_m_audit():
    res = by_setting(load("p4_dev_system2.json")["runs"],
                     lambda r: (r["setting"]["interval"], r["setting"]["m"], r["setting"]["audit"]))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    marker = {500: "o", 750: "s", 1000: "^"}
    for (iv, m, a), (mm, rel) in res.items():
        ax.scatter(mm, rel, marker=marker[m], s=70, zorder=3, color=BLUE if iv == 14 else GOLD)
    for m, mk in marker.items():
        ax.scatter([], [], marker=mk, color=GRAY, label=f"m = {m}")
    ax.scatter([], [], color=BLUE, label="retrain every 14 d"); ax.scatter([], [], color=GOLD, label="retrain every 7 d")
    ax.annotate("chosen: 14 d, m 500, audit 10 %", (0.92, 8.12), xytext=(10, 6), textcoords="offset points", color=GREEN)
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, 12)
    limit(ax, 5)
    ax.set_title("Threshold-set size m and audit share (tuned forest), development half")
    save(fig, "docs/img/p4/system_m_audit.png")


def final_test_windows():
    r = load("p5_final_test.json")["runs"]
    n = len(r[0]["windows"])
    miss = [np.mean([100 * x["windows"][i]["missed"] / x["windows"][i]["td"] for x in r]) for i in range(n)]
    rel = [np.mean([100 * x["windows"][i]["released"] / x["windows"][i]["fc"] for x in r]) for i in range(n)]
    td = [r[0]["windows"][i]["td"] for i in range(n)]
    labels = [f"window {i + 1}\n{td[i]} true defects" for i in range(n)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    a1.bar(labels, rel, color=BLUE, width=0.55)
    a1.set_ylabel("False calls released (%)"); a1.set_title("Released per 14-day window")
    a2.bar(labels, miss, color=[RED if m > 1 else NAVY for m in miss], width=0.55)
    a2.axhline(1, color=RED, ls="--", lw=1.5)
    a2.set_ylabel("True defects missed (%)"); a2.set_title("Missed per 14-day window (limit 1 %)")
    for ax, v in ((a1, rel), (a2, miss)):
        for i, x in enumerate(v):
            ax.text(i, x, f"{x:.1f}" if ax is a1 else f"{x:.2f}", ha="center", va="bottom")
        ax.tick_params(axis="x", labelsize=10)
    fig.suptitle("Final test (data days 64–101): frozen model, retrained every 14 days, mean of 5 seeds")
    save(fig, "docs/img/p5/final_test_windows.png")


def dev_ideas():
    res = {}
    for f in ("p6_dev_ideas.json", "p6_dev_ideas2.json", "p6_dev_spc.json"):
        res |= by_setting(load(f)["runs"], lambda r: r["idea"], recs="days")
    show = {"frozen": "frozen model", "burst x3": "burst retrain", "threshold-set cap 30/day": "cap 30/day",
            "same-day switch 30": "same-day switch 30", "spc day-total q99": "SPC switch",
            "spc day-total q99 + cap 30 + burst x3": "SPC + cap + burst retrain (tested)",
            "cap 30 + switch 30 + burst x3": "switch + cap + burst retrain",
            "spc day-total q99 + cap 30 + switch 30 + burst x3": "all four"}
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for k, lab in show.items():
        mm, rel = res[k]
        c = GREEN if "tested" in lab else (NAVY if k == "frozen" else (BLUE if mm <= 1 else GRAY))
        ax.scatter(mm, rel, s=80, color=c, zorder=3)
        ax.annotate(lab, (mm, rel), xytext=(7, 3), textcoords="offset points", color=c, fontsize=10)
    ax.set_ylim(0, 13)
    limit(ax, 3)
    ax.set_title("Improvement ideas, day-by-day simulation, development half only")
    save(fig, "docs/img/p6/dev_ideas.png")


def second_evaluation():
    p5, p6 = load("p5_final_test.json")["runs"], load("p6_final_test.json")["runs"]
    m5 = [np.mean([x["slices"][i]["missed_pct"] for x in p5]) for i in range(5)]
    m6 = [np.mean([x["slices"][i]["missed_pct"] for x in p6]) for i in range(5)]
    x = np.arange(5); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, m5, w, color=NAVY, label="P5 frozen model (first evaluation)")
    ax.bar(x + w / 2, m6, w, color=GOLD, label="SPC + cap + burst retrain (second evaluation)")
    ax.axhline(1, color=RED, ls="--", lw=1.5)
    ax.set_xticks(x, [f"slice {i + 1}" for i in range(5)])
    ax.set_ylabel("True defects missed (%)"); ax.legend(frameon=False)
    ax.set_title("Test half per slice of [7]: the improvements did not carry over")
    save(fig, "docs/img/p6/second_evaluation.png")


if __name__ == "__main__":
    forest_tuning(); system_m_audit(); final_test_windows(); dev_ideas(); second_evaluation()
