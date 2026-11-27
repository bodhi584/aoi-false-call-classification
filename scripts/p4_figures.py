"""P4 tables and figures from the development-half experiments.

Inputs:  results/p4_dev_runs.json, results/p4_dev_rules.json, data/dataset.csv
Outputs: results/p4_dev_summary.json, results/p4_dev_summary.md, docs/img/p4/*.png
Metrics per period of >= 299 true defects: missed = released true defects / true defects,
released = released false calls / false calls. A setting passes if missed <= 1 % in every period and
every seed; among passing settings the one with the highest released share is chosen.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import capped_latest  # noqa: E402

NAVY, BLUE, GOLD, RED, GRAY = "#14213D", "#2F6DB5", "#B8870A", "#BF382B", "#8A93A0"
IMG = Path("docs/img/p4")


def name(s):
    it = "once" if s["interval"] is None else f"every {s['interval']} d"
    w = "all" if s["window"] is None else f"{s['window']} d"
    cap = f", cap {s['cap']}/day" if s.get("cap") else ""
    return f"{s['rule']}, retrain {it}, window {w}, m {s['m']}, audit {int(100 * s['audit'])} %{cap}"


def summarise(runs):
    rows = {}
    for r in runs:
        rows.setdefault(json.dumps(r["setting"], sort_keys=True), []).append(r)
    out = []
    for key, rs in rows.items():
        P = len(rs[0]["periods"])
        miss = np.array([[p["missed_pct"] for p in r["periods"]][:P] for r in rs])   # seeds x periods
        rel = np.array([[p["released_pct"] for p in r["periods"]][:P] for r in rs])
        tot = [100 * sum(p["released"] for p in r["periods"]) / sum(p["fc"] for p in r["periods"]) for r in rs]
        out.append({"setting": json.loads(key), "name": name(json.loads(key)), "seeds": len(rs),
                    "missed_pct_mean": miss.mean(0).round(2).tolist(), "missed_pct_max": miss.max(0).round(2).tolist(),
                    "released_pct_mean": rel.mean(0).round(2).tolist(), "released_total_pct": round(float(np.mean(tot)), 2),
                    "passes": bool((miss <= 1.0).all()), "worst_missed_pct": round(float(miss.max()), 2)})
    return sorted(out, key=lambda r: (not r["passes"], -r["released_total_pct"]))


def burst_figure(path, start, m=299):
    """True defects per calendar day; the threshold set of the first step and the first check window."""
    df = pd.read_csv(path, usecols=["timestamp", "class", "meta_feat1"])
    date = pd.to_datetime(df["timestamp"]).dt.normalize()
    cal = (date - date.iloc[0]).dt.days.to_numpy()
    n = len(df) // 2
    y = df["class"].to_numpy()[:n]; cal = cal[:n]
    per_day = pd.Series(y).groupby(cal).sum()
    td = np.where(y == 1)[0]
    thr = capped_latest(td[cal[td] < start], cal, m, None)
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=200)
    ax.bar(per_day.index, per_day.values, color=GRAY, width=0.9, label="true defects per day")
    tday = pd.Series(cal[thr]).value_counts()
    ax.bar(tday.index, tday.values, color=GOLD, width=0.9, label=f"threshold set of step 1 (latest {m})")
    ax.axvspan(start, start + 7, color=BLUE, alpha=0.15, label="check window of step 1")
    ax.axvline(start, color=NAVY, lw=1.5); ax.text(start, ax.get_ylim()[1] * 0.92, " T0", color=NAVY)
    ax.set_xlabel("calendar day (development half)"); ax.set_ylabel("true defects")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(IMG / "burst_threshold_set.png")
    m1 = df["meta_feat1"].to_numpy()[:n]
    chk = td[(cal[td] >= start) & (cal[td] < start + 7)]
    return {"threshold_set_days": {int(k): int(v) for k, v in tday.sort_index().items()},
            "threshold_set_top_meta_feat1": {int(k): int(v) for k, v in pd.Series(m1[thr]).value_counts().head(3).items()},
            "check_window_top_meta_feat1": {int(k): int(v) for k, v in pd.Series(m1[chk]).value_counts().head(3).items()},
            "check_window_true_defects": int(len(chk))}


def scatter(summary, fname, title, label=False):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    col = {"np": GOLD, "plain": BLUE, "cv": RED, "half": GRAY}
    for r in summary:
        s = r["setting"]
        ax.scatter(max(r["worst_missed_pct"], 0.05), r["released_total_pct"], s=40,
                   color=col[s["rule"]], marker="o" if s["interval"] else "x", alpha=0.85)
        if label:
            txt = s["rule"] + (f", cap {s['cap']}/day" if s.get("cap") else "")
            ax.annotate(txt, (max(r["worst_missed_pct"], 0.05), r["released_total_pct"]), fontsize=8,
                        xytext=(5, 4), textcoords="offset points")
    ax.axvline(1, color=NAVY, ls="--", lw=1); ax.text(1.05, ax.get_ylim()[1] * 0.95, "1 % limit", color=NAVY, fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("worst missed share over periods and seeds (%), log scale")
    ax.set_ylabel("false calls released (%)"); ax.set_title(title, fontsize=10)
    for k, c in col.items():
        if any(r["setting"]["rule"] == k for r in summary):
            ax.scatter([], [], color=c, label=k)
    ax.scatter([], [], color=GRAY, marker="x", label="no retraining")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(IMG / fname)


def main(path="data/dataset.csv"):
    IMG.mkdir(parents=True, exist_ok=True)
    runs = json.loads(Path("results/p4_dev_runs.json").read_text())
    out = {"start_day": runs["start_day"], "grid": summarise(runs["runs"])}
    scatter(out["grid"], "dev_grid.png", "Experiment 1: 40 settings x 5 seeds, development half")
    if Path("results/p4_dev_rules.json").exists():
        rules = json.loads(Path("results/p4_dev_rules.json").read_text())
        out["rules"] = summarise(rules["runs"])
        scatter(out["rules"], "dev_rules.png", "Experiment 2: threshold rules, weekly retraining, m = 299", label=True)
    out["burst"] = burst_figure(path, runs["start_day"])
    Path("results/p4_dev_summary.json").write_text(json.dumps(out, indent=2))
    lines = ["| setting | missed % per period (mean / max over seeds) | released % per period | released total % | passes |",
             "|---|---|---|---|---|"]
    for part in ("grid", "rules"):
        for r in out.get(part, []):
            lines.append(f"| {r['name']} | {r['missed_pct_mean']} / {r['missed_pct_max']} | {r['released_pct_mean']} "
                         f"| {r['released_total_pct']} | {'yes' if r['passes'] else 'no'} |")
    Path("results/p4_dev_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main(*sys.argv[1:])
