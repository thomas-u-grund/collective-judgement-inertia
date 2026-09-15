"""
Matched persistence measure for Polymarket (clean, >1 day before resolution):
recent_rate (rolling mean of own is_contrarian over past 20 decisions)
correlated, within-wallet (demeaned Pearson r), with the decision exactly
K steps ahead -- same construction as sim/analyze_habit.py (Manifold) and
the GJP persistence_matched.py script.
"""
import time
import numpy as np
import pandas as pd
from scipy import stats

RECENT_WINDOW = 20

t0 = time.time()
d = pd.read_parquet(
    "poly_decisions_panel.parquet",
    columns=["wallet", "seq", "timestamp", "end_date", "is_contrarian", "n_resolved_so_far"],
)
d["gap_to_resolution"] = d["end_date"] - d["timestamp"]
q = d[(d["n_resolved_so_far"] >= 2) & (d["gap_to_resolution"] > 86400)].dropna(subset=["is_contrarian"]).copy()
q = q.sort_values(["wallet", "seq"]).reset_index(drop=True)
print(f"qualifying n={len(q):,} wallets={q['wallet'].nunique():,} [{time.time()-t0:.1f}s]")

n = len(q)
recent_rate = np.full(n, np.nan, dtype="float64")
exact_k = {K: np.full(n, np.nan, dtype="float64") for K in [1, 2, 5, 10, 25, 50, 100, 250, 500]}

for wallet, idx in q.groupby("wallet", sort=False).indices.items():
    idx = np.asarray(idx)
    c = q["is_contrarian"].values[idx].astype(float)
    m = len(c)
    cum = np.concatenate([[0.0], np.cumsum(c)])
    pos = np.arange(m)
    start = np.clip(pos - RECENT_WINDOW, 0, None)
    win_n = pos - start
    rate = np.where(win_n > 0, (cum[pos] - cum[start]) / np.maximum(win_n, 1), np.nan)
    recent_rate[idx] = rate
    for K, arr in exact_k.items():
        valid = pos + K < m
        out = np.full(m, np.nan)
        out[valid] = c[pos[valid] + K]
        arr[idx] = out

q["recent_rate"] = recent_rate
for K, arr in exact_k.items():
    q[f"exact_{K}"] = arr
print(f"features built [{time.time()-t0:.1f}s]")


def within_person_r(dd, x, y):
    dd = dd.dropna(subset=[x, y]).copy()
    dd[f"{x}_dm"] = dd[x] - dd.groupby("wallet")[x].transform("mean")
    dd[f"{y}_dm"] = dd[y] - dd.groupby("wallet")[y].transform("mean")
    r, p = stats.pearsonr(dd[f"{x}_dm"], dd[f"{y}_dm"])
    return r, p, len(dd)


qr = q.dropna(subset=["recent_rate"])
for K in [1, 2, 5, 10, 25, 50, 100, 250, 500]:
    r, p, nn = within_person_r(qr, "recent_rate", f"exact_{K}")
    print(f"K={K:>4d}  r={r:+.5f}  p={p:.2e}  n={nn:,}  [{time.time()-t0:.1f}s]")
