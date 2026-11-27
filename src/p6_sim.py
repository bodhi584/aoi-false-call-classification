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
  - intraday: safety switch within the day: once the operator has confirmed `intraday` true defects today,
    nothing more is released for the rest of the day (assumes operator labels come back in line, in row order)
Only labels that exist in production are used: every joint sent to the operator, plus the audited share
of the released joints, labelled at the end of its day.
"""
import numpy as np

from p4_sim import MIN_TRAIN_TD, N_NUM, capped_latest, make_forest, threshold


def simulate_daily(X, y, cal_day, n_dev, *, start_day, interval, m, audit, seed, forest=None,
                   burst=None, burst_min=20, audit_trigger=False, pause=0, cap=None, unseen=False,
                   intraday=None, first_row=0, n_end=None, out_rel=None):
    rng = np.random.default_rng(seed)
    n = n_end or n_dev
    idx = np.arange(n)
    cd = cal_day[:n]
    known = (cd < start_day) | (idx < first_row)
    days = np.unique(cd[(cd >= start_day) & (idx >= first_row)])
    model, t, seen, next_sched, pending, pause_left = None, -np.inf, None, start_day, None, 0
    daily_td, records = [], []
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
            thr_set = capped_latest(td, cal_day, m, cap)
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
        records.append({"T": int(d), "td": int(yc.sum()), "fc": int((yc == 0).sum()),
                        "missed": int((rel & (yc == 1)).sum()), "released": int((rel & (yc == 0)).sum()),
                        "t": float(t), "retrain": why, "trigger": trig})
    return records
