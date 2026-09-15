import time
import numpy as np
import pandas as pd

K_MIN_KNOWN = 10
MIN_BASELINE = 20
MIN_QUALIFYING_ROWS = 20
RECENT_WINDOW = 20

t0 = time.time()
df = pd.read_parquet(
    "poly_decisions_panel.parquet",
    columns=["wallet", "seq", "timestamp", "end_date", "correct", "is_contrarian", "n_resolved_so_far"],
)
print(f"full sample: n={len(df):,} wallets={df['wallet'].nunique():,} [{time.time()-t0:.1f}s]")

n = len(df)
recent_skill = np.full(n, np.nan)
baseline_skill = np.full(n, np.nan)
n_known_prior = np.zeros(n, dtype="int32")
baseline_n = np.zeros(n, dtype="int32")

for wallet, idx in df.groupby("wallet", sort=False).indices.items():
    idx = np.asarray(idx)
    sub_t = df["timestamp"].values[idx]
    sub_end = df["end_date"].values[idx]
    sub_c = df["correct"].values[idx].astype(float)
    order_res = np.argsort(sub_end, kind="stable")
    res_sorted = sub_end[order_res]
    c_sorted = sub_c[order_res]
    cum = np.concatenate([[0.0], np.cumsum(c_sorted)])
    n_known = np.searchsorted(res_sorted, sub_t, side="right")
    window_start = np.clip(n_known - RECENT_WINDOW, 0, None)
    recent_n = n_known - window_start
    rs = np.where(recent_n > 0, (cum[n_known] - cum[window_start]) / np.maximum(recent_n, 1), np.nan)
    bn = window_start
    bs = np.where(bn > 0, cum[window_start] / np.maximum(bn, 1), np.nan)
    recent_skill[idx] = rs
    baseline_skill[idx] = bs
    n_known_prior[idx] = n_known
    baseline_n[idx] = bn

df["recent_skill"] = recent_skill
df["baseline_skill"] = baseline_skill
df["n_known_prior"] = n_known_prior
df["baseline_n"] = baseline_n
print(f"features built [{time.time()-t0:.1f}s]")

q = df[(df["n_known_prior"] >= K_MIN_KNOWN) & (df["baseline_n"] >= MIN_BASELINE)].dropna(
    subset=["recent_skill", "baseline_skill", "is_contrarian"]
).copy()
counts = q.groupby("wallet").size()
keep = counts[counts >= MIN_QUALIFYING_ROWS].index
q = q[q["wallet"].isin(keep)].copy()
print(f"qualifying: n={len(q):,} wallets={q['wallet'].nunique():,} [{time.time()-t0:.1f}s]")
q.to_parquet("poly_decisions_panel_matched_full.parquet", index=False)
