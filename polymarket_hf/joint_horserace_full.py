import time
import numpy as np
import pandas as pd

RECENT_WINDOW = 20


def cluster_robust_ols(d, xcols, ycol, cluster_col):
    X = np.column_stack([d[f"{c}_dm"].values for c in xcols] + [np.ones(len(d))])
    y = d[f"{ycol}_dm"].values
    XtX_inv = np.linalg.inv(X.T @ X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    codes = pd.factorize(d[cluster_col].values)[0]
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    Xs = X[order]; us = resid[order]
    boundaries = np.flatnonzero(np.diff(codes_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(codes_sorted)]])
    for s, e in zip(starts, ends):
        Xg = Xs[s:e]; ug = us[s:e]
        score_g = Xg.T @ ug
        meat += np.outer(score_g, score_g)
    n_clusters = len(starts)
    n, k = X.shape
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    cov = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, se, n_clusters, n


def demean(d, cols, id_col="wallet"):
    d = d.copy()
    means = d.groupby(id_col)[cols].transform("mean")
    for c in cols:
        d[f"{c}_dm"] = d[c] - means[c]
    return d


def report(label, beta, se, n_clusters, n, xcols):
    print(f"\n-- {label} --  n={n:,}  wallets={n_clusters:,}")
    for i, c in enumerate(xcols):
        b, s = beta[i], se[i]
        t = b / s
        lo, hi = b - 1.96 * s, b + 1.96 * s
        print(f"  {c:>22s}: beta={b:+.5f}  se={s:.5f}  t={t:+.2f}  95% CI [{lo:+.5f}, {hi:+.5f}]")


t0 = time.time()
skill = pd.read_parquet("poly_decisions_panel_matched_full.parquet")
print(f"loaded: {skill.shape} [{time.time()-t0:.1f}s]")
skill = skill.sort_values(["wallet", "seq"]).reset_index(drop=True)
n = len(skill)
recent_rate = np.full(n, np.nan)
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
skill["recent_contrarian_rate"] = recent_rate
print(f"recent_contrarian_rate built [{time.time()-t0:.1f}s]")

q = skill.dropna(subset=["recent_skill", "baseline_skill", "recent_contrarian_rate", "is_contrarian"])
print(f"final joint sample: n={len(q):,} wallets={q['wallet'].nunique():,}")

d = demean(q, ["recent_skill", "baseline_skill", "recent_contrarian_rate", "is_contrarian"])
beta, se, nc, nn = cluster_robust_ols(
    d, ["recent_skill", "baseline_skill", "recent_contrarian_rate"], "is_contrarian", "wallet"
)
report("Polymarket FULL SAMPLE joint model", beta, se, nc, nn,
       ["recent_skill", "baseline_skill", "recent_contrarian_rate"])
print(f"[{time.time()-t0:.1f}s]")
