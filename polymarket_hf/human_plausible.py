import time
import numpy as np
import pandas as pd
from scipy import stats

RECENT_WINDOW = 20
t0 = time.time()

mg = pd.read_parquet("wallet_median_gaps.parquet")
human_wallets = set(mg.loc[mg["median_gap_s"] >= 60, "wallet"])
print(f"human-plausible wallets (median gap >= 60s): {len(human_wallets):,} / {len(mg):,}")

skill = pd.read_parquet("poly_decisions_panel_matched.parquet")
skill = skill[skill["wallet"].isin(human_wallets)].copy()
print(f"human-plausible joint-model sample: n={len(skill):,} wallets={skill['wallet'].nunique():,} [{time.time()-t0:.1f}s]")

skill = skill.sort_values(["wallet", "seq"]).reset_index(drop=True)
n = len(skill)
recent_rate = np.full(n, np.nan)
exact_k = {K: np.full(n, np.nan) for K in [1, 10, 50, 100]}
for wallet, idx in skill.groupby("wallet", sort=False).indices.items():
    idx = np.asarray(idx)
    c = skill["is_contrarian"].values[idx].astype(float)
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
skill["recent_contrarian_rate"] = recent_rate
for K, arr in exact_k.items():
    skill[f"exact_{K}"] = arr
print(f"features built [{time.time()-t0:.1f}s]")


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


def demean(d, cols, id_col="wallet"):
    d = d.copy()
    means = d.groupby(id_col)[cols].transform("mean")
    for c in cols:
        d[f"{c}_dm"] = d[c] - means[c]
    return d


q = skill.dropna(subset=["recent_skill", "baseline_skill", "recent_contrarian_rate", "is_contrarian"])
print(f"joint sample: n={len(q):,} wallets={q['wallet'].nunique():,}")
xcols = ["recent_skill", "baseline_skill", "recent_contrarian_rate"]
d = demean(q, xcols + ["is_contrarian"])
beta, se, nc, nn = cluster_robust_ols(d, xcols, "is_contrarian", "wallet")
print(f"\nHuman-plausible joint model: n={nn:,} wallets={nc:,}")
for i, c in enumerate(xcols):
    b, s = beta[i], se[i]
    t = b / s
    print(f"  {c:>22s}: beta={b:+.5f}  se={s:.5f}  t={t:+.2f}  95% CI [{b-1.96*s:+.5f}, {b+1.96*s:+.5f}]")

print("\nPersistence (exact-K), human-plausible sample:")
qr = skill.dropna(subset=["recent_contrarian_rate"])
for K in [1, 10, 50, 100]:
    sub = qr.dropna(subset=[f"exact_{K}"]).copy()
    if sub["wallet"].nunique() < 10:
        print(f"K={K}: too few")
        continue
    sub["x_dm"] = sub["recent_contrarian_rate"] - sub.groupby("wallet")["recent_contrarian_rate"].transform("mean")
    sub["y_dm"] = sub[f"exact_{K}"] - sub.groupby("wallet")[f"exact_{K}"].transform("mean")
    r, p = stats.pearsonr(sub["x_dm"], sub["y_dm"])
    print(f"K={K:>4d}  r={r:+.5f}  p={p:.3e}  n={len(sub):,}  wallets={sub['wallet'].nunique():,}")
print(f"[{time.time()-t0:.1f}s]")
