"""All decision tables of P4 (model selection), P5 (final test) and P6 (improvement attempts), from the result files.

Output: docs/results_tables.md (every number in docs/results_story.md comes from this file).
Definitions: missed % = released true defects / true defects; released % = released false calls / false calls;
a period = consecutive data with >= 299 true defects; "worst missed" = maximum over all periods and all seeds;
"released" = released false calls / false calls over the whole run, mean of the seeds.
A setting PASSES if worst missed <= 1 %. Among passing settings the highest released share wins.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import load  # noqa: E402

R = Path("results")
OUT = []


def j(name):
    return json.loads((R / name).read_text())


def table(title, head, rows, note=None):
    OUT.append(f"\n### {title}\n")
    if note:
        OUT.append(note + "\n")
    OUT.append("| " + " | ".join(head) + " |")
    OUT.append("|" + "---|" * len(head))
    for r in rows:
        OUT.append("| " + " | ".join(str(c) for c in r) + " |")


def f2(x):
    return f"{x:.2f}"


def summarise(runs, key, recs="windows"):
    g = defaultdict(list)
    for r in runs:
        rel = 100 * sum(w["released"] for w in r[recs]) / sum(w["fc"] for w in r[recs])
        g[key(r)].append((max(p["missed_pct"] for p in r["periods"]), rel,
                          np.mean([p["missed_pct"] for p in r["periods"]])))
    out = []
    for k, v in g.items():
        out.append((k, max(a for a, _, _ in v), np.mean([b for _, b, _ in v]), min(b for _, b, _ in v),
                    max(b for _, b, _ in v), np.mean([c for _, _, c in v]), len(v)))
    return sorted(out, key=lambda r: (r[1] > 1, -r[2]))


def rows_of(s, chosen=None):
    return [[("**" + str(k) + "**") if k == chosen else k, f2(mm), "yes" if mm <= 1 else "no", f2(rel),
             f"{f2(lo)}–{f2(hi)}", f2(mean_m), n] for k, mm, rel, lo, hi, mean_m, n in s]


HEAD = ["setting", "worst missed %", "passes ≤ 1 %", "released % (mean)", "released % (min–max seeds)",
        "missed % (mean of periods)", "seeds"]


def sys_name(s):
    it = "train once" if s["interval"] is None else f"every {s['interval']} d"
    w = "all history" if s.get("window") is None else f"last {s['window']} d"
    cap = f", cap {s['cap']}/day" if s.get("cap") else ""
    grp = ", one threshold per inspection type" if s.get("per_type") or s.get("groups") else ""
    return f"{s['rule'].upper()}, {it}, {w}, m {s['m']}, audit {int(100 * s['audit'])} %{cap}{grp}"


def forest_name(f):
    samp = "1:1" if f["sampling_strategy"] == "all" else f"ratio {f['sampling_strategy']}"
    return f"{f['n_estimators']} trees, max_features {f['max_features']}, min leaf {f['min_samples_leaf']}, {samp}"


def data_tables():
    d, p = j("data_facts.json"), j("p4_data_prep.json")
    X, y, cal_day, n_dev = load("data/dataset.csv")
    table("D1 Data facts", ["fact", "value", "source"], [
        ["joints (AOI calls)", f"{d['joints']:,}", "data_facts.json"],
        ["boards", f"{d['boards']:,}", "data_facts.json"],
        ["days with data / calendar days", f"{d['days_with_data']} / {d['calendar_days']}", "data_facts.json"],
        ["true defects (joints)", f"{d['true_defect_joints']:,} ({d['true_defect_share_of_joints_pct']} %)", "data_facts.json"],
        ["columns: measurements + categorical", f"{p['measurements']} + {p['categorical']}", "p4_data_prep.json"],
        ["inputs after one-hot", p["inputs_after_one_hot"], "p4_data_prep.json"],
        ["missing values", p["missing_values"], "p4_data_prep.json"],
        ["constant measurements", len(p["constant_measurements"]), "p4_data_prep.json"],
        ["duplicate rows (kept)", f"{p['duplicate_rows']:,}", "p4_data_prep.json"],
        ["component codes (meta_feat1) only in test half", p["meta_feat1_codes_only_in_test_half"], "p4_data_prep.json"],
        ["drift: AUROC first vs last 30 % of days", d["drift_test_auroc_first_vs_last_30pct_days"], "data_facts.json"],
    ])
    n = len(y)
    table("D2 Split (same as [7]: first half of the rows / second half)", ["part", "rows", "data days", "calendar days",
          "true defects", "false calls", "use"], [
        ["development", f"{n_dev:,}", "1–64", f"{cal_day[0]}–{cal_day[n_dev - 1]}", f"{int(y[:n_dev].sum()):,}",
         f"{int((y[:n_dev] == 0).sum()):,}", "all design, tuning and selection"],
        ["test", f"{n - n_dev:,}", "64–101", f"{cal_day[n_dev]}–{cal_day[-1]}", f"{int(y[n_dev:].sum()):,}",
         f"{int((y[n_dev:] == 0).sum()):,}", "P5 final test (once); P6 second evaluation (once)"],
    ], note="Day 64 is split between the halves (row order). In the test, all development rows are labelled history.")
    k = {m: None for m in (299, 500, 750, 1000)}
    from p4_sim import np_rank
    table("D3 Neyman–Pearson rank k (≤ 1 % missed with 95 % confidence) [14, Alg. 1]",
          ["threshold-set size m", "k (k-th lowest score is the threshold)", "plain 1 % quantile rank"],
          [[m, np_rank(m), int(np.floor(0.01 * m)) + 1] for m in k])


def p4_tables():
    runs = j("p4_dev_runs.json")["runs"]
    table("P4-1 Experiment 1: system parameters S1–S3, S5 (default forest), 40 settings × 5 seeds", HEAD,
          rows_of(summarise(runs, lambda r: sys_name(r["setting"]))),
          note="Development half, T0 = calendar day 40, periods from 14-day windows. Source: results/p4_dev_runs.json")
    table("P4-2 Experiment 2: threshold rules (weekly retraining, all history, m 299, audit 100 %)", HEAD,
          rows_of(summarise(j("p4_dev_rules.json")["runs"], lambda r: sys_name(r["setting"]))),
          note="CV = method of [7] (5-fold, recall ≥ 0.99, mean); HALF = classical 0.5 cut. Source: results/p4_dev_rules.json")
    table("P4-3 Experiment 3: day cap and per-type thresholds (observation only, not in the P3 plan)", HEAD,
          rows_of(summarise(j("p4_dev_cap.json")["runs"], lambda r: sys_name(r["setting"]))),
          note="Source: results/p4_dev_cap.json")
    best = forest_name({"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "sampling_strategy": "all"})
    table("P4-4 Experiment 4: forest parameters F1–F4 on the chosen system (14 d, all history, m 500, audit 10 %)", HEAD,
          rows_of(summarise(j("p4_dev_forest.json")["runs"], lambda r: forest_name(r["setting"]["forest"])), best),
          note="Source: results/p4_dev_forest.json")
    table("P4-5 Experiment 5: system re-check with the tuned forest", HEAD,
          rows_of(summarise(j("p4_dev_confirm.json")["runs"], lambda r: sys_name(r["setting"]))),
          note="Source: results/p4_dev_confirm.json")
    table("P4-6 Experiment 6: threshold-set size m (S3) and audit share (S5) with the tuned forest", HEAD,
          rows_of(summarise(j("p4_dev_system2.json")["runs"], lambda r: sys_name(r["setting"]))),
          note="Source: results/p4_dev_system2.json")
    table("P4-9 Experiment 7 (added 28 Nov, after the final test): model choice and confidence S4, same system", HEAD,
          rows_of(summarise(j("p4_dev_models.json")["runs"], lambda r: r["model"]), "BRF (frozen)"),
          note="Every 14 d, all history, m 500, audit 10 %, NP 95 % unless stated; only the model changes. WRF = RF with "
               "class_weight balanced_subsample. NP 90 % gives the same rank k = 2 as 95 % for m = 500 (identical to "
               "the frozen model). Development half only; does not change the final-test result. "
               "Source: results/p4_dev_models.json")
    fi = j("p4_feature_importance.json")
    top = list(fi["all"]["importance"].items())[:10]
    table("P4-7 Feature importance (permutation, drop in average precision; one-hot groups permuted together)",
          ["rank", "feature", "importance"], [[i + 1, k, f"{v:.3f}"] for i, (k, v) in enumerate(top)],
          note=f"Fit on the first 80 % of the development half, measured on the last 20 % ({fi['rows_measure']:,} rows, "
               f"{fi['true_defects_measure']} true defects); average precision of the model {fi['all']['average_precision']:.3f}. "
               "Source: results/p4_feature_importance.json")
    rows = []
    for t in sorted(k for k in fi if k.startswith("type_")):
        it = list(fi[t]["importance"].items())[:3]
        rows.append([t.replace("type_", ""), fi[t].get("true_defects", ""), f"{fi[t]['average_precision']:.3f}",
                     ", ".join(f"{k} ({v:.3f})" for k, v in it)])
    table("P4-8 Feature importance per inspection type", ["inspection type", "true defects (measure set)",
          "average precision", "top 3 features"], rows)


def eval_tables(name, label):
    r = j(name)
    R_ = r["runs"]
    ref = r["ref7_brfc2"]
    m = lambda f: np.mean([f(x) for x in R_])  # noqa: E731
    tot = ["missed_pct", "released_pct", "recall", "precision", "f1"]
    table(f"{label}-1 Whole test half, per seed", ["seed", "missed %", "released %", "recall", "precision", "F1",
          "true defects", "false calls"],
          [[x["seed"]] + [f2(x["total"][k]) if k.endswith("pct") else f"{x['total'][k]:.4f}" for k in tot]
           + [x["total"]["td"], x["total"]["fc"]] for x in R_]
          + [["**mean**"] + [f2(m(lambda x, k=k: x["total"][k])) if k.endswith("pct") else
                             f"{m(lambda x, k=k: x['total'][k]):.4f}" for k in tot] + ["", ""]])
    table(f"{label}-2 Per slice of [7] (5 equal chronological row slices of the test half)",
          ["slice", "true defects", "missed % mean", "missed % max (seeds)", "released % mean",
           "[7] BRFC2 missed %", "[7] BRFC2 released %"],
          [[i + 1, R_[0]["slices"][i]["td"], f2(m(lambda x, i=i: x["slices"][i]["missed_pct"])),
            f2(max(x["slices"][i]["missed_pct"] for x in R_)), f2(m(lambda x, i=i: x["slices"][i]["released_pct"])),
            ref["missed"][i], ref["released"][i]] for i in range(5)],
          note="[7], Sect. VI, Table IV, model BRFC2, mean of 10 runs.")
    np_ = min(len(x["periods"]) for x in R_)
    table(f"{label}-3 Per period (≥ 299 true defects)", ["period", "true defects", "missed % mean",
          "missed % max (seeds)", "released % mean"],
          [[i + 1, R_[0]["periods"][i]["td"], f2(m(lambda x, i=i: x["periods"][i]["missed_pct"])),
            f2(max(x["periods"][i]["missed_pct"] for x in R_)), f2(m(lambda x, i=i: x["periods"][i]["released_pct"]))]
           for i in range(np_)])


def p5_windows():
    R_ = j("p5_final_test.json")["runs"]
    n = len(R_[0]["windows"])
    table("P5-4 Per 14-day window (one retraining each)", ["window", "start (calendar day)", "true defects",
          "false calls", "training true defects", "threshold t (mean)", "missed % mean", "released % mean"],
          [[i + 1, R_[0]["windows"][i]["T"], R_[0]["windows"][i]["td"], R_[0]["windows"][i]["fc"],
            int(np.mean([x["windows"][i]["train_td"] for x in R_])), f"{np.mean([x['windows'][i]['t'] for x in R_]):.3f}",
            f2(np.mean([100 * x["windows"][i]["missed"] / x["windows"][i]["td"] for x in R_])),
            f2(np.mean([100 * x["windows"][i]["released"] / x["windows"][i]["fc"] for x in R_]))] for i in range(n)])


def p6_tables():
    runs = []
    for f in ("p6_dev_ideas.json", "p6_dev_ideas2.json", "p6_dev_spc.json"):
        runs += j(f)["runs"]
    table("P6-1 Improvement ideas on the development half (day-by-day simulation, 5 seeds each)", HEAD,
          rows_of(summarise(runs, lambda r: r["idea"], recs="days"), "spc day-total q99 + cap 30 + burst x3"),
          note="Periods pooled from daily records (finer than the 14-day windows of P4, so the frozen model shows "
               "2.30 % here). burst xK: extra retraining next morning if today's labelled true defects ≥ K × mean of the "
               "last 14 production days and ≥ 20. same-day switch N: no release for the rest of the day after N "
               "operator-confirmed true defects. cap 30: at most 30 threshold-set true defects per day. spc day-total "
               "qQ: no release while today's AOI calls exceed the Q quantile of the previous 14 production days. "
               "Sources: results/p6_dev_ideas.json, p6_dev_ideas2.json, p6_dev_spc.json")


def burst_tables():
    X, y, cal_day, n_dev = load("data/dataset.csv")
    import pandas as pd
    from p6_spc import spc_alarms
    al = spc_alarms("data/dataset.csv", mode="day", q=0.99)
    df = pd.DataFrame({"d": cal_day, "y": y, "a": al, "test": np.arange(len(y)) >= n_dev})
    g = df.groupby(["test", "d"]).agg(td=("y", "sum"), calls=("y", "size"), alarm=("a", "mean")).reset_index()
    p5 = j("p5_final_test.json")["runs"]; p6 = j("p6_final_test.json")["runs"]
    missed5 = defaultdict(list)
    big = g[g.td >= 60].sort_values(["test", "d"])
    rows = []
    for _, r in big.iterrows():
        med = g[(g.test == r.test) & (g.d < r.d) & (g.d >= r.d - 14)].calls.median()
        rows.append(["test" if r.test else "development", int(r.d), int(r.td), f"{int(r.calls):,}",
                     f"{r.calls / med:.1f}" if med == med else "", f"{100 * r.alarm:.0f}"])
    table("B1 Burst days (≥ 60 true defects on one calendar day)", ["half", "calendar day", "true defects",
          "AOI calls", "calls / median of previous 14 days", "share of calls in SPC alarm %"], rows,
          note="Descriptive only (no tuning). The development bursts on days 41 and 57 came with 3.4× and 10.3× the median call "
               "volume; the largest test bursts (days 119, 126, 127) came with only 0.7–2.5×, so the SPC switch did not "
               "fire. Days before calendar day 40 are before deployment (no switch active).")
    for name, lab in (("p5_final_test.json", "P5 frozen model"), ("p6_final_test.json", "P6 improved system")):
        R_ = j(name)["runs"]
        if "days" in R_[0]:
            dm = defaultdict(list)
            for x in R_:
                for d in x["days"]:
                    dm[d["T"]].append(d["missed"])
            tot = sum(np.mean(v) for v in dm.values())
            top = sorted(dm.items(), key=lambda kv: -np.mean(kv[1]))[:5]
            table(f"B2 Where the misses of the {lab} fall (test half, mean of 5 seeds)",
                  ["calendar day", "true defects that day", "missed (mean)", "share of all misses %"],
                  [[d, int(g[(g.test) & (g.d == d)].td.iloc[0]), f"{np.mean(v):.1f}", f"{100 * np.mean(v) / tot:.0f}"]
                   for d, v in top])


def business_table():
    X, y, cal_day, n_dev = load("data/dataset.csv")
    days = len(np.unique(cal_day[n_dev:]))
    rows = []
    for name, lab in (("p5_final_test.json", "P5 frozen model"), ("p6_final_test.json", "P6 improved system")):
        R_ = j(name)["runs"]
        rel = np.mean([x["total"]["tn"] for x in R_]); miss = np.mean([x["total"]["fn"] for x in R_])
        fc = R_[0]["total"]["fc"]; td = R_[0]["total"]["td"]
        rows.append([lab, f"{fc:,}", f"{rel:,.0f}", f"{rel / days:,.0f}", f"{fc - rel:,.0f}", f"{td:,}",
                     f"{miss:.1f}", f"{miss / days:.2f}"])
    table("C1 What the system does on the test half (counts, mean of 5 seeds)",
          ["system", "false calls", "released (no manual check)", "released per production day",
           "false calls still checked", "true defects", "true defects missed", "missed per production day"], rows,
          note=f"Test half: {days} production days (calendar days with data). Operator time per joint is not in the "
               "data, so no hours are given.")


def main():
    OUT.append("# Results tables (generated by scripts/make_results_tables.py — do not edit by hand)")
    OUT.append(__doc__.split("\n", 2)[2])
    OUT.append("\n## Data and split")
    data_tables()
    OUT.append("\n## P4 model selection (development half only)")
    p4_tables()
    OUT.append("\n## P5 final test (test half, frozen model, run once)")
    eval_tables("p5_final_test.json", "P5")
    p5_windows()
    OUT.append("\n## P6 improvement attempts (development half) and second evaluation (test half, run once)")
    p6_tables()
    eval_tables("p6_final_test.json", "P6E")
    business_table()
    OUT.append("\n## Bursts (descriptive)")
    burst_tables()
    Path("docs/results_tables.md").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
