import time
import numpy as np
import pandas as pd
from scipy import stats

RECENT_WINDOW = 20
t0 = time.time()
q = pd.read_parquet("poly_decisions_panel_matched.parquet", columns=["wallet", "seq", "timestamp", "is_contrarian"])
raw = pd.read_parquet("poly_decisions_panel.parquet", columns=["wallet", "seq", "market_id"])
q = q.merge(raw, on=["wallet", "seq"], how="left")
q = q.sort_values(["wallet", "timestamp"])
first_touch = q.drop_duplicates(subset=["wallet", "market_id"], keep="first").copy()
first_touch = first_touch.sort_values(["wallet", "timestamp"])
first_touch["seq2"] = first_touch.groupby("wallet").cumcount() + 1
print(f"distinct-market panel: n={len(first_touch):,} wallets={first_touch['wallet'].nunique():,} (from {len(q):,} total) [{time.time()-t0:.1f}s]")

n = len(first_touch)
recent_rate = np.full(n, np.nan)
exact_k = {K: np.full(n, np.nan) for K in [1, 5, 10, 25, 50, 100]}
for wallet, idx in first_touch.groupby("wallet", sort=False).indices.items():
    idx = np.asarray(idx)
    c = first_touch["is_contrarian"].values[idx].astype(float)
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
first_touch["recent_rate"] = recent_rate
for K, arr in exact_k.items():
    first_touch[f"exact_{K}"] = arr
print(f"features built [{time.time()-t0:.1f}s]")

def within_person_r(dd, x, y):
    dd = dd.dropna(subset=[x, y]).copy()
    dd[f"{x}_dm"] = dd[x] - dd.groupby("wallet")[x].transform("mean")
    dd[f"{y}_dm"] = dd[y] - dd.groupby("wallet")[y].transform("mean")
    r, p = stats.pearsonr(dd[f"{x}_dm"], dd[f"{y}_dm"])
    return r, p, len(dd)

qr = first_touch.dropna(subset=["recent_rate"])
for K in [1, 5, 10, 25, 50, 100]:
    sub = qr.dropna(subset=[f"exact_{K}"])
    if sub["wallet"].nunique() < 10:
        print(f"K={K}: too few wallets")
        continue
    r, p, nn = within_person_r(qr, "recent_rate", f"exact_{K}")
    print(f"K={K:>4d}  r={r:+.5f}  p={p:.3e}  n={nn:,}  wallets={sub['wallet'].nunique():,}  [{time.time()-t0:.1f}s]")
