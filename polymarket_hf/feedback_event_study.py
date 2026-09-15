"""
Resolution-feedback event study for Polymarket, matching Manifold's exact
construction: for each decision, the outcome (correct/incorrect) of that
wallet's most recently resolved prior decision strictly before the current
decision's timestamp, then joint model with recent contrarian rate.
Primary (bot-excluded, >1 day before resolution) specification.
"""
import time
import numpy as np
import pandas as pd

RECENT_WINDOW = 20
t0 = time.time()

d = pd.read_parquet(
    "poly_decisions_panel.parquet",
    columns=["wallet", "seq", "timestamp", "end_date", "correct", "is_contrarian"],
)
d["gap_to_resolution"] = d["end_date"] - d["timestamp"]
df = d[d["gap_to_resolution"] > 86400].copy()
df = df.sort_values(["wallet", "timestamp"]).reset_index(drop=True)
df["seq2"] = df.groupby("wallet").cumcount() + 1
print(f"clean sample: n={len(df):,} wallets={df['wallet'].nunique():,} [{time.time()-t0:.1f}s]")

n = len(df)
last_resolved_correct = np.full(n, np.nan)
recent_rate = np.full(n, np.nan)

for wallet, idx in df.groupby("wallet", sort=False).indices.items():
    idx = np.asarray(idx)
    t = df["timestamp"].values[idx]
    end = df["end_date"].values[idx]
    c = df["correct"].values[idx].astype(float)
    ic = df["is_contrarian"].values[idx].astype(float)
    m = len(idx)

    order_end = np.argsort(end, kind="stable")
    end_sorted = end[order_end]
    c_sorted = c[order_end]
    # for each t_i, find the resolved-prior decision with the LARGEST end_date < t_i
    pos = np.searchsorted(end_sorted, t, side="left")  # count of end_dates < t_i
    for k in range(m):
        p = pos[k]
        if p > 0:
            last_resolved_correct[idx[k]] = c_sorted[p - 1]

    pos2 = np.arange(m)
    start = np.clip(pos2 - RECENT_WINDOW, 0, None)
    cum = np.concatenate([[0.0], np.cumsum(ic)])
    win_n = pos2 - start
    rate = np.where(win_n > 0, (cum[pos2] - cum[start]) / np.maximum(win_n, 1), np.nan)
    recent_rate[idx] = rate

df["last_resolved_correct"] = last_resolved_correct
df["recent_contrarian_rate"] = recent_rate
print(f"features built [{time.time()-t0:.1f}s]")

q = df.dropna(subset=["last_resolved_correct", "recent_contrarian_rate", "is_contrarian"])
print(f"qualifying: n={len(q):,} wallets={q['wallet'].nunique():,}")


def cluster_robust_ols(dd, xcols, ycol, cluster_col):
    X = np.column_stack([dd[f"{c}_dm"].values for c in xcols] + [np.ones(len(dd))])
    y = dd[f"{ycol}_dm"].values
    XtX_inv = np.linalg.inv(X.T @ X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    codes = pd.factorize(dd[cluster_col].values)[0]
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
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return beta, se, n_clusters, nn, r2


def demean(dd, cols, id_col="wallet"):
    dd = dd.copy()
    means = dd.groupby(id_col)[cols].transform("mean")
    for c in cols:
        dd[f"{c}_dm"] = dd[c] - means[c]
    return dd


def report(label, beta, se, nc, nn, r2, xcols):
    print(f"\n-- {label} --  n={nn:,}  wallets={nc:,}  within-R2={r2:.5f}")
    for i, c in enumerate(xcols):
        b, s = beta[i], se[i]
        t = b / s
        print(f"  {c:>22s}: beta={b:+.5f}  se={s:.5f}  t={t:+.2f}  95% CI [{b-1.96*s:+.5f}, {b+1.96*s:+.5f}]")


d1 = demean(q, ["last_resolved_correct", "is_contrarian"])
beta, se, nc, nn, r2 = cluster_robust_ols(d1, ["last_resolved_correct"], "is_contrarian", "wallet")
report("last_resolved_correct alone", beta, se, nc, nn, r2, ["last_resolved_correct"])

d2 = demean(q, ["recent_contrarian_rate", "is_contrarian"])
beta, se, nc, nn, r2 = cluster_robust_ols(d2, ["recent_contrarian_rate"], "is_contrarian", "wallet")
report("recent_contrarian_rate alone", beta, se, nc, nn, r2, ["recent_contrarian_rate"])

d3 = demean(q, ["last_resolved_correct", "recent_contrarian_rate", "is_contrarian"])
beta, se, nc, nn, r2 = cluster_robust_ols(d3, ["last_resolved_correct", "recent_contrarian_rate"], "is_contrarian", "wallet")
report("joint model", beta, se, nc, nn, r2, ["last_resolved_correct", "recent_contrarian_rate"])
print(f"[{time.time()-t0:.1f}s]")
