"""P6 ideas: day-by-day simulation of the second check with special retraining triggers.

Same system as the frozen P4 model (src/p4_sim.py), but the clock moves one calendar day at a time,
so the system can react to what the labels of the day show:
  - scheduled retraining every `interval` days (as before)
  - burst trigger: if the true defects labelled today are >= `burst` x the mean of the last 14 production
    days (and at least `burst_min`), retrain tomorrow morning
  - audit trigger: if the random audit finds a released true defect today, retrain tomorrow morning
  - pause: after a trigger, release nothing for `pause` production days
  - cap: at most `cap` threshold-set true defects from one calendar day
  - unseen: a joint with a category code that never occurred in the training data is never released
  - alarm: bool per row from an SPC chart on the AOI calls (src/p6_spc.py); a joint in alarm is never released
  - comp, comp_switch: per-component p-chart within the day. For each component code, the operator-confirmed true
    defects today are compared with n * p + k * sqrt(n * p * (1 - p)) (k = comp_switch), where n = joints of that
    component sent to the operator so far today and p = its true-defect rate among labelled joints of the previous
    14 production days. Once a component is above the limit (and has >= comp_min confirmed true defects today),
    no joint of that component is released for the rest of the day. Other components are not affected.
  - comp_cap: at most comp_cap threshold-set true defects from one component code
  - intraday: safety switch within the day: once the operator has confirmed `intraday` true defects today,
    nothing more is released for the rest of the day (assumes operator labels come back in line, in row order)
Only labels that exist in production are used: every joint sent to the operator, plus the audited share
of the released joints, labelled at the end of its day.
"""
import numpy as np

from p4_sim import MIN_TRAIN_TD, N_NUM, capped_latest, make_forest, threshold


def simulate_daily(X, y, cal_day, n_dev, *, start_day, interval, m, audit, seed, forest=None,
                   burst=None, burst_min=20, audit_trigger=False, pause=0, cap=None, unseen=False,
                   intraday=None, alarm=None, comp=None, comp_switch=None, comp_min=5,
                   comp_cap=None, first_row=0, n_end=None, out_rel=None):
    rng = np.random.default_rng(seed)
    n = n_end or n_dev
    idx = np.arange(n)
    cd = cal_day[:n]
    known = (cd < start_day) | (idx < first_row)
    days = np.unique(cd[(cd >= start_day) & (idx >= first_row)])
    model, t, seen, next_sched, pending, pause_left = None, -np.inf, None, start_day, None, 0
    daily_td, records, prod_days = [], [], []
    for d in days:
        # ---- morning: retrain if scheduled or triggered yesterday
        why = None
        if d >= next_sched:                                  # fixed calendar: T0, T0 + 14, T0 + 28, ...
            why = "schedule"
            while next_sched <= d:
                next_sched += interval
        elif pending:
            why = pending
        if why:
            hist = idx[((cd < d) | (idx < first_row)) & known]
            td = hist[y[hist] == 1]
            thr_set = capped_latest(td, cal_day, m, cap) if comp_cap is None else comp_capped_latest(td, comp, m, comp_cap)
            train = np.setdiff1d(hist, thr_set)
            if int(y[train].sum()) >= MIN_TRAIN_TD:
                model = make_forest(seed, forest).fit(X[train], y[train])
                t = threshold(model.predict_proba(X[thr_set])[:, 1], "np")
                seen = X[train][:, N_NUM:].sum(0) > 0
            pending = None
        # ---- the day: score, release, operator labels
        chk = idx[(cd == d) & (idx >= first_row)]
        if model is None or pause_left > 0:
            rel = np.zeros(len(chk), bool)
        else:
            rel = model.predict_proba(X[chk])[:, 1] < t
            if unseen:
                rel &= ~((X[chk][:, N_NUM:] > 0) & ~seen).any(1)
            if alarm is not None:                            # SPC alarm on the AOI call stream: send to operator
                rel &= ~alarm[chk]
            if comp_switch is not None:
                rel = component_switch(rel, chk, y, comp, known, cd, prod_days[-14:], comp_switch, comp_min)
            if intraday:                                     # confirmed true defects before each joint today
                confirmed = np.cumsum((~rel) & (y[chk] == 1)) - ((~rel) & (y[chk] == 1))
                stop = np.flatnonzero(confirmed >= intraday)
                if len(stop):
                    rel[stop[0]:] = False
        pause_left = max(pause_left - 1, 0)
        if out_rel is not None:
            out_rel[chk] = rel
        yc = y[chk]
        known[chk[~rel]] = True
        aud = rng.random(int(rel.sum())) < audit
        known[chk[rel]] = aud
        # ---- evening: what the labels of today show
        td_today = int((yc[~rel] == 1).sum() + (yc[rel][aud] == 1).sum())
        audit_hit = bool((yc[rel][aud] == 1).any())
        base = np.mean(daily_td[-14:]) if daily_td else None
        trig = None
        if burst and base is not None and td_today >= max(burst * base, burst_min):
            trig = "burst"
        if audit_trigger and audit_hit:
            trig = "audit"
        if trig:
            pending = trig
            pause_left = pause
        daily_td.append(td_today)
        prod_days.append(d)
        records.append({"T": int(d), "td": int(yc.sum()), "fc": int((yc == 0).sum()),
                        "missed": int((rel & (yc == 1)).sum()), "released": int((rel & (yc == 0)).sum()),
                        "t": float(t), "retrain": why, "trigger": trig})
    return records


def comp_capped_latest(td_idx, comp, m, cap):
    """Latest m true defects, but at most cap from one component code (keeps the threshold set diverse)."""
    out, per = [], {}
    for i in td_idx[::-1]:
        c = comp[i]
        if per.get(c, 0) < cap:
            out.append(i); per[c] = per.get(c, 0) + 1
            if len(out) == m:
                break
    return np.sort(np.array(out, dtype=int))


def component_switch(rel, chk, y, comp, known, cd, past_days, k, min_td):
    """Per-component p-chart, joint by joint in time order; returns the new release decisions."""
    rel = rel.copy()
    lab = known & np.isin(cd, past_days) if len(past_days) else known & (cd < cd[chk[0]])
    ncomp = int(comp.max()) + 1
    n_lab = np.bincount(comp[:len(cd)][lab], minlength=ncomp)
    n_td = np.bincount(comp[:len(cd)][lab & (y[:len(cd)] == 1)], minlength=ncomp)
    p_all = n_td.sum() / max(n_lab.sum(), 1)
    p = np.where(n_lab >= 50, n_td / np.maximum(n_lab, 1), p_all)          # few labels: use the overall rate
    p = np.clip(p, 1e-4, 0.999)
    sent, conf, stopped = np.zeros(ncomp), np.zeros(ncomp), np.zeros(ncomp, bool)
    cc, yy = comp[chk], y[chk]
    for j in range(len(chk)):
        c = cc[j]
        if stopped[c]:
            rel[j] = False
        if not rel[j]:                                     # sent to the operator: label known at once
            sent[c] += 1
            if yy[j] == 1:
                conf[c] += 1
                ucl = sent[c] * p[c] + k * np.sqrt(sent[c] * p[c] * (1 - p[c]))
                if conf[c] >= min_td and conf[c] > ucl:
                    stopped[c] = True
    return rel
