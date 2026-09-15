"""
Continuous performance measure for Polymarket, replacing binary
side-correctness with the probability implied by the purchased token's
price for the outcome that actually occurred -- a continuous performance
score, not a Brier-loss transform (1-Brier is a distinct, monotonically
related quantity) -- in the identical rolling-20/expanding-baseline
windowing as the main text, primary (near-resolution-excluded)
specification.
"""
import time
import numpy as np
import pandas as pd

K_MIN_KNOWN = 10
MIN_BASELINE = 20
MIN_QUALIFYING_ROWS = 20
RECENT_WINDOW = 20

t0 = time.time()
d = pd.read_parquet(
    "poly_decisions_panel.parquet",
    columns=["wallet", "seq", "timestamp", "end_date", "price", "correct", "is_contrarian", "n_resolved_so_far"],
)
d["gap_to_resolution"] = d["end_date"] - d["timestamp"]
df = d[d["gap_to_resolution"] > 86400].copy()
df["prob_true"] = np.where(df["correct"] == 1, df["price"], 1 - df["price"])
print(f"clean sample: n={len(df):,} wallets={df['wallet'].nunique():,} [{time.time()-t0:.1f}s]")

n = len(df)
recent_skill = np.full(n, np.nan)
baseline_skill = np.full(n, np.nan)
n_known_prior = np.zeros(n, dtype="int32")
baseline_n = np.zeros(n, dtype="int32")

for wallet, idx in df.groupby("wallet", sort=False).indices.items():
    idx = np.asarray(idx)
    sub_t = df["timestamp"].values[idx]
    sub_end = df["end_date"].values[idx]
    sub_s = df["prob_true"].values[idx].astype(float)
    order_res = np.argsort(sub_end, kind="stable")
    res_sorted = sub_end[order_res]
    s_sorted = sub_s[order_res]
    cum = np.concatenate([[0.0], np.cumsum(s_sorted)])
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

df["recent_skill_brier"] = recent_skill
df["baseline_skill_brier"] = baseline_skill
df["n_known_prior"] = n_known_prior
df["baseline_n"] = baseline_n
print(f"features built [{time.time()-t0:.1f}s]")

q = df[(df["n_known_prior"] >= K_MIN_KNOWN) & (df["baseline_n"] >= MIN_BASELINE)].dropna(
    subset=["recent_skill_brier", "baseline_skill_brier", "is_contrarian"]
).copy()
counts = q.groupby("wallet").size()
keep = counts[counts >= MIN_QUALIFYING_ROWS].index
q = q[q["wallet"].isin(keep)].copy()
print(f"qualifying: n={len(q):,} wallets={q['wallet'].nunique():,} [{time.time()-t0:.1f}s]")

q = q.sort_values(["wallet", "seq"]).reset_index(drop=True)
n = len(q)
recent_rate = np.full(n, np.nan)
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
q["recent_contrarian_rate"] = recent_rate
q = q.dropna(subset=["recent_contrarian_rate"])
print(f"final: n={len(q):,} wallets={q['wallet'].nunique():,} [{time.time()-t0:.1f}s]")


def cluster_robust_ols(d, xcols, ycol, cluster_col):
    X = np.column_stack([d[f"{c}_dm"].values for c in xcols] + [np.ones(len(d))])
    y = d[f"{ycol}_dm"].values
    XtX_inv = np.linalg.inv(X.T @ X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    codes = pd.factorize(d[cluster_col].values)[0]
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    Xs = X[order]; us = resid[order]
    boundaries = np.flatnonzero(np.diff(codes_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(codes_sorted)]])
    meat = np.zeros((X.shape[1], X.shape[1]))
    for s, e in zip(starts, ends):
        Xg = Xs[s:e]; ug = us[s:e]
        sc = Xg.T @ ug
        meat += np.outer(sc, sc)
    n_clusters = len(starts)
    nn, k = X.shape
    correction = (n_clusters / (n_clusters - 1)) * ((nn - 1) / (nn - k))
    cov = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, se, n_clusters, nn


def demean(d, cols, id_col):
    d = d.copy()
    means = d.groupby(id_col)[cols].transform("mean")
    for c in cols:
        d[f"{c}_dm"] = d[c] - means[c]
    return d


xcols = ["recent_skill_brier", "baseline_skill_brier", "recent_contrarian_rate"]
dd = demean(q, xcols + ["is_contrarian"], "wallet")
beta, se, nc, nn = cluster_robust_ols(dd, xcols, "is_contrarian", "wallet")
print(f"\nPolymarket Brier-based joint model (primary spec): n={nn:,} wallets={nc:,}")
for i, c in enumerate(xcols):
    b, s = beta[i], se[i]
    t = b / s
    print(f"  {c:>22s}: beta={b:+.5f}  se={s:.5f}  t={t:+.2f}  95% CI [{b-1.96*s:+.5f}, {b+1.96*s:+.5f}]")
print(f"[{time.time()-t0:.1f}s]")
