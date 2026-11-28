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


def proof_bars(res, chosen, title, path, labels=None, note=None):
    """One evidence figure: released share (left) and worst-period missed (right) per alternative, chosen in green."""
    names = sorted(res, key=lambda k: (k != chosen, res[k][0] > 1, -res[k][1]))
    if chosen is None:                                  # no winner shown: baseline first, then by result
        names = sorted(res, key=lambda k: ("frozen" not in k, res[k][0] > 1, -res[k][1]))
    lab = [labels.get(k, k) if labels else k for k in names]
    rel = [res[k][1] for k in names]; mm = [res[k][0] for k in names]
    h = max(3.2, 0.42 * len(names) + 1.6)
    w = 12.5 + max(0, max(len(x) for x in lab) - 32) * 0.11
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    yy = np.arange(len(names))[::-1]
    a1.barh(yy, rel, color=[GREEN if k == chosen else (BLUE if res[k][0] <= 1 else GRAY) for k in names], height=0.62)
    a1.set_yticks(yy, lab); a1.set_xlabel("False calls released (%)  — higher is better")
    a2.barh(yy, [max(m, 0.02) for m in mm], color=[GREEN if k == chosen else (RED if m > 1 else NAVY)
                                                     for k, m in zip(names, mm)], height=0.62)
    a2.axvline(1, color=RED, ls="--", lw=1.5); a2.set_xscale("symlog", linthresh=1)
    a2.set_xlim(0, max(max(mm) * 1.6, 2))
    a2.set_xlabel("Worst period: true defects missed (%)  — must be ≤ 1 %")
    for ax, v in ((a1, rel), (a2, mm)):
        for i, x in zip(yy, v):
            ax.text(max(x, 0.02), i, f" {x:.2f}", va="center", fontsize=10)
    fig.suptitle(title, fontsize=13)
    if note:
        fig.text(0.01, 0.005, note, fontsize=9, color=GRAY)
    save(fig, path)


def evidence_figures():
    P = "docs/img/evidence/"
    mod = by_setting(load("p4_dev_models.json")["runs"], lambda r: r["model"])
    proof = by_setting(load("p4_dev_proof.json")["runs"], lambda r: r["name"])
    conf = by_setting(load("p4_dev_confirm.json")["runs"],
                      lambda r: (r["setting"]["interval"], r["setting"]["window"]))
    # 1 model
    proof_bars({k: v for k, v in mod.items() if k != "BRF, NP 99 %"}, "BRF (frozen)",
               "1  Model: only the balanced random forest holds 1 % (same system, 5 seeds)", P + "1_model.png",
               {"BRF (frozen)": "BRF (chosen)"}, "Development half (days 1–64). Every 14 d, all history, m 500, NP 95 %, audit 10 %; only the model changes.")
    # 2 threshold rule and confidence
    thr = {"NP 95 % (chosen)": mod["BRF (frozen)"], "NP 99 %": mod["BRF, NP 99 %"],
           "plain 1 % quantile": proof["threshold: plain 1 % quantile"],
           "[7] cross-validated threshold": proof["threshold: [7] cross-validated"],
           "classical 0.5 cut": proof["threshold: 0.5 cut"]}
    proof_bars(thr, "NP 95 % (chosen)", "2  Threshold rule: only Neyman–Pearson keeps 1 %; 95 % releases twice as much as 99 %",
               P + "2_threshold.png", note="Frozen system and BRF. NP 90 % gives the same threshold rank (k = 2) as 95 % for m = 500, so it is identical.")
    # 3 retraining schedule and window
    ret = {"every 14 d, all history (chosen)": conf[(14, None)], "every 7 d, all history": conf[(7, None)],
           "every 14 d, last 28 d": conf[(14, 28)], "every 7 d, last 28 d": conf[(7, 28)],
           "train once": proof["retraining: train once"], "every 28 d, all history": proof["retraining: every 28 d"]}
    proof_bars(ret, "every 14 d, all history (chosen)",
               "3  Retraining: every 14 days with all history releases most while keeping 1 %", P + "3_retraining.png",
               note="Frozen BRF, m 500, NP 95 %, audit 10 %.")
    # 4 threshold-set size and audit share (14-day schedule)
    sm = by_setting(load("p4_dev_system2.json")["runs"],
                    lambda r: (r["setting"]["interval"], r["setting"]["m"], r["setting"]["audit"]))
    ma = {f"m {m}, audit {int(100 * a)} %" + (" (chosen)" if (m, a) == (500, 0.1) else ""): v
          for (iv, m, a), v in sm.items() if iv == 14}
    proof_bars(ma, "m 500, audit 10 % (chosen)", "4  Threshold-set size m and audit share (every 14 d)",
               P + "4_m_audit.png", note="Frozen BRF. All 9 combinations with 7-day retraining fail (worst 1.63–4.60 %).")
    # 5 forest parameters
    fo = by_setting(load("p4_dev_forest.json")["runs"], lambda r: tuple(sorted(r["setting"]["forest"].items())))
    def fn(k):
        f = dict(k); samp = "1:1" if f["sampling_strategy"] == "all" else f"ratio {f['sampling_strategy']}"
        return f"{f['n_estimators']} trees, {f['max_features']}, leaf {f['min_samples_leaf']}, {samp}"
    fo = {fn(k) + (" (chosen)" if fn(k) == "300 trees, sqrt, leaf 5, 1:1" else ""): v for k, v in fo.items()}
    proof_bars(fo, "300 trees, sqrt, leaf 5, 1:1 (chosen)", "5  Forest parameters (24 settings): 300 trees, sqrt, leaf 5, 1:1",
               P + "5_forest.png", note="Every 14 d, all history, m 500, NP 95 %, audit 10 %.")
    # 6 periodic vs triggered retraining (day-by-day simulation)
    ideas = {}
    for f in ("p6_dev_ideas.json", "p6_dev_ideas2.json", "p6_dev_spc.json"):
        ideas |= by_setting(load(f)["runs"], lambda r: r["idea"], recs="days")
    keep = {"frozen": "periodic only (frozen model)", "burst x3": "+ burst-triggered retraining",
            "audit trigger": "+ audit-triggered retraining", "spc day-total q99": "+ SPC switch",
            "threshold-set cap 30/day": "+ day cap 30", "same-day switch 30": "+ same-day switch",
            "spc day-total q99 + cap 30 + burst x3": "+ SPC + cap + burst retraining",
            "cap 30 + switch 30 + burst x3": "+ same-day switch + cap + burst retraining"}
    proof_bars({v: ideas[k] for k, v in keep.items()}, None,
               "6  Periodic vs triggered retraining and switches (development half, day-by-day periods)",
               P + "6_triggers.png", note="Finer day-by-day periods. The combinations reach 0 % here but did not hold on the test half (second evaluation: 1.20 % missed).")


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


def component_figure():
    res = by_setting(load("p7_dev_component.json")["runs"], lambda r: r["idea"], recs="days")
    g = {}
    for f in ("p6_dev_ideas.json", "p6_dev_spc.json"):
        g |= by_setting(load(f)["runs"], lambda r: r["idea"], recs="days")
    res["global SPC on calls (P6)"] = g["spc day-total q99"]
    res["global SPC + day cap 30 + burst retraining (P6)"] = g["spc day-total q99 + cap 30 + burst x3"]
    res["frozen model"] = res.pop("frozen")
    proof_bars(res, None, "7  Component-code monitoring (development half, day-by-day periods)",
               "docs/img/evidence/7_component.png",
               note="Component switch = same-day p-chart of confirmed true defects per component; component SPC = AOI calls of one "
                    "component today vs its 99 % quantile; cap = per component in the threshold set.")


FROZEN = {"interval": 14, "window": None, "m": 500, "rule": "np", "audit": 0.1,
          "n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "sampling_strategy": "all"}


def all_window_runs():
    """Every 14-day-window run on the development half with its full parameter set."""
    out = []
    for f in ("p4_dev_forest.json", "p4_dev_confirm.json", "p4_dev_system2.json", "p4_dev_proof.json", "p4_dev_sweep.json"):
        for r in load(f)["runs"]:
            st = dict(r["setting"]); fo = st.pop("forest", None) or r.get("forest") or {}
            fo = {k: v for k, v in fo.items() if k != "model"}
            if "model" in (r.get("forest") or {}):
                continue
            prm = dict(FROZEN) | {k: st[k] for k in ("interval", "window", "m", "rule", "audit") if k in st} | fo
            if r.get("setting", {}).get("cap") or r.get("setting", {}).get("per_type"):
                continue
            rel = 100 * sum(w["released"] for w in r["windows"]) / sum(w["fc"] for w in r["windows"])
            out.append((prm, r["seed"], max(p["missed_pct"] for p in r["periods"]), rel))
    return out


def tuning_curves():
    runs = all_window_runs()
    mf = lambda v: {"sqrt": 14, "log2": 7}.get(v, int(v * 206) if isinstance(v, float) else v)  # noqa: E731
    specs = [("min_samples_leaf", "F3 min samples per leaf", float, "log"),
             ("n_estimators", "F1 number of trees", float, "linear"),
             ("max_features", "F2 features per split (of 206)", mf, "log"),
             ("m", "S3 threshold-set size m", float, "linear"),
             ("interval", "S1 retraining interval (days)", float, "linear"),
             ("audit", "S5 audit share (%)", lambda v: 100 * v, "linear")]
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), gridspec_kw={"height_ratios": [1.3, 1, 1.3, 1]})
    summary = {}
    for i, (key, label, conv, scale) in enumerate(specs):
        top, bot = axes[2 * (i // 3), i % 3], axes[2 * (i // 3) + 1, i % 3]
        bot.sharex(top)
        g = defaultdict(list)
        for prm, seed, mm, rel in runs:
            if all(prm[k] == FROZEN[k] for k in FROZEN if k != key) and prm[key] is not None:
                g[prm[key]].append((mm, rel))
        xs = sorted(g, key=conv)
        X = [conv(v) for v in xs]
        rel_m = [np.mean([b for _, b in g[v]]) for v in xs]
        rel_lo = [min(b for _, b in g[v]) for v in xs]; rel_hi = [max(b for _, b in g[v]) for v in xs]
        worst = [max(a for a, _ in g[v]) for v in xs]
        ok = [w <= 1 for w in worst]
        top.fill_between(X, rel_lo, rel_hi, color=BLUE, alpha=0.15, lw=0)
        top.plot(X, rel_m, color=BLUE, lw=2)
        top.scatter(X, rel_m, s=[55 if o else 35 for o in ok], color=[BLUE if o else GRAY for o in ok], zorder=3,
                    edgecolor="white")
        c = X[xs.index(FROZEN[key])]
        top.scatter([c], [rel_m[xs.index(FROZEN[key])]], marker="*", s=320, color=GREEN, zorder=4, edgecolor="white")
        top.set_title(label, fontsize=12); top.set_ylabel("released %" if i % 3 == 0 else "")
        bot.plot(X, worst, color=NAVY, lw=1.6)
        bot.scatter(X, worst, color=[NAVY if o else RED for o in ok], s=30, zorder=3)
        bot.axhline(1, color=RED, ls="--", lw=1.3); bot.set_ylabel("worst missed %" if i % 3 == 0 else "")
        bot.set_ylim(0, max(2.5, min(max(worst) * 1.1, 6)))
        if scale == "log":
            top.set_xscale("log")
        bot.set_xticks(X, [str(round(x)) for x in X]); bot.minorticks_off(); top.minorticks_off()
        top.tick_params(labelbottom=False)
        bot.tick_params(axis="x", labelsize=9)
        summary[key] = [{"value": v, "released": round(r, 2), "worst_missed": round(w, 2), "passes": o}
                        for v, r, w, o in zip(xs, rel_m, worst, ok)]
    fig.suptitle("Tuning curves around the frozen model (development half, 5 seeds; one parameter changed at a time)\n"
                 "blue line = mean released (band = seeds), grey point = fails 1 %; lower panel = worst period missed; "
                 "★ = frozen value", fontsize=12)
    save(fig, "docs/img/evidence/8_tuning_curves.png")
    Path("results/p4_tuning_curves.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    forest_tuning(); evidence_figures(); component_figure(); tuning_curves(); system_m_audit(); final_test_windows(); dev_ideas(); second_evaluation()
