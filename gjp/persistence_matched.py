"""
Recompute the persistence test on GJP using the EXACT same construction as
the Manifold headline measure (sim/analyze_habit.py): a rolling recent-rate
(mean is_contrarian over the past X=20 decisions) correlated, within person
(demeaned Pearson r), with the decision EXACTLY K steps ahead (not a
lag-K autocorrelation of the raw series -- a related but different design).
"""
import numpy as np
import pandas as pd
from scipy import stats

RECENT_WINDOW = 20


def add_recent_rate(df, id_col, seq_col, y_col, X=RECENT_WINDOW):
    df = df.sort_values([id_col, seq_col])
    out = []
    for _, d in df.groupby(id_col, sort=False):
        c = d[y_col].values.astype(float)
        n = len(c)
        cum = np.concatenate([[0.0], np.cumsum(c)])
        idx = np.arange(n)
        start = np.clip(idx - X, 0, None)
        win_n = idx - start
        rate = np.where(win_n > 0, (cum[idx] - cum[start]) / np.maximum(win_n, 1), np.nan)
        d = d.copy()
        d["recent_rate"] = rate
        out.append(d)
    return pd.concat(out, ignore_index=True)


def add_exact_k(df, id_col, seq_col, y_col, K):
    df = df.sort_values([id_col, seq_col])
    out = []
    for _, d in df.groupby(id_col, sort=False):
        c = d[y_col].values.astype(float)
        n = len(c)
        exact = np.full(n, np.nan)
        valid = np.arange(n) + K < n
        exact[valid] = c[np.arange(n)[valid] + K]
        d = d.copy()
        d[f"exact_{K}"] = exact
        out.append(d)
    return pd.concat(out, ignore_index=True)


def within_person_r(d, id_col, x, y):
    dd = d.dropna(subset=[x, y]).copy()
    dd[f"{x}_dm"] = dd[x] - dd.groupby(id_col)[x].transform("mean")
    dd[f"{y}_dm"] = dd[y] - dd.groupby(id_col)[y].transform("mean")
    r, p = stats.pearsonr(dd[f"{x}_dm"], dd[f"{y}_dm"])
    return r, p, len(dd)


if __name__ == "__main__":
    decisions = pd.read_parquet("gjp_decisions_panel.parquet")
    decisions = add_recent_rate(decisions, "user_id", "seq", "is_contrarian")
    q = decisions.dropna(subset=["recent_rate"])
    print(f"n={len(q):,} users={q.user_id.nunique():,}")
    for K in [1, 2, 5, 10, 25, 50, 100]:
        qk = add_exact_k(q, "user_id", "seq", "is_contrarian", K)
        r, p, n = within_person_r(qk, "user_id", "recent_rate", f"exact_{K}")
        print(f"K={K:>4d}  r={r:+.5f}  p={p:.2e}  n={n:,}")
