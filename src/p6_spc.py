"""SPC alarms on the AOI call stream: label-free, known the moment each joint is called.

For each row j: count the AOI calls in the 60 minutes up to and including its timestamp
(all calls, or only calls of the same component code meta_feat1). The row is in alarm if the count is
above the upper control limit: the q-quantile (default 99 %) of the same rolling counts over the previous
14 production days (past data only; at least 50 past calls, otherwise no alarm).
"""
import numpy as np
import pandas as pd


def spc_alarms(path, by_code=False, window="60min", days_back=14, min_ucl=10, q=0.99, mode="rolling"):
    df = pd.read_csv(path, usecols=["timestamp", "meta_feat1"])           # same row order as load()
    ts = pd.to_datetime(df["timestamp"])
    key = df["meta_feat1"].astype(str) if by_code else pd.Series("all", index=df.index)
    day = ts.dt.normalize()

    # 1) rolling 60-minute call count of each row, within its key
    count = np.empty(len(df))
    ones = pd.Series(1.0, index=ts)
    for _, pos in key.groupby(key).indices.items():
        if mode == "rolling":                        # calls of the key in the last 60 minutes
            count[pos] = ones.iloc[pos].rolling(window).sum().to_numpy()
        else:                                        # "day": calls of the key so far today
            count[pos] = pd.Series(1.0, index=pos).groupby(day.to_numpy()[pos]).cumsum().to_numpy()

    # 2) control limit per key and day = q-quantile of the rolling counts of that key over the previous
    #    14 production days (empirical limit: AOI calls come in clusters per board, so a Poisson limit is too tight)
    days = np.sort(day.unique())
    drow = np.searchsorted(days, day.to_numpy())
    ucl = np.full(len(df), np.inf)
    for _, pos in key.groupby(key).indices.items():
        d_k, c_k = drow[pos], count[pos]
        for i in np.unique(d_k):
            past = c_k[(d_k >= i - days_back) & (d_k < i)]
            if len(past) >= 50:
                ucl[pos[d_k == i]] = max(np.quantile(past, q), min_ucl)
    return count > ucl
