import time
import numpy as np
import pandas as pd


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
    Xs = X[order]
    us = resid[order]
    boundaries = np.flatnonzero(np.diff(codes_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(codes_sorted)]])
    for s, e in zip(starts, ends):
        Xg = Xs[s:e]
        ug = us[s:e]
        score_g = Xg.T @ ug
        meat += np.outer(score_g, score_g)
    n_clusters = len(starts)
    n, k = X.shape
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    cov = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, se, n_clusters, n


def demean(d, cols):
    d = d.copy()
    means = d.groupby("wallet")[cols].transform("mean")
    for c in cols:
        d[f"{c}_dm"] = d[c] - means[c]
    return d


def report(label, beta, se, n_clusters, n, xcols):
    print(f"\n-- {label} --  n={n:,}  wallets={n_clusters:,}")
    for i, c in enumerate(xcols):
        b, s = beta[i], se[i]
        t = b / s
        lo, hi = b - 1.96 * s, b + 1.96 * s
        print(f"  {c:>18s}: beta={b:+.5f}  se={s:.5f}  t={t:+.2f}  95% CI [{lo:+.5f}, {hi:+.5f}]")


t0 = time.time()
decisions = pd.read_parquet(
    "poly_decisions_panel.parquet",
    columns=["wallet", "seq", "price", "is_contrarian", "recent_accuracy", "n_resolved_so_far"],
)
decisions["wallet"] = decisions["wallet"].astype("category")
print("loaded:", decisions.shape, "wallets:", decisions["wallet"].nunique(), f"[{time.time()-t0:.1f}s]")

qualifying = decisions[decisions["n_resolved_so_far"] >= 2].dropna(
    subset=["recent_accuracy", "is_contrarian"]
).copy()
print("qualifying:", len(qualifying), "wallets:", qualifying["wallet"].nunique(), f"[{time.time()-t0:.1f}s]")

print("\n" + "=" * 70)
print("TEST A: recent accuracy -> departure from consensus (wallet FE)")
print("=" * 70)
d = demean(qualifying, ["recent_accuracy", "is_contrarian"])
print(f"demeaned [{time.time()-t0:.1f}s]")
beta, se, nc, n = cluster_robust_ols(d, ["recent_accuracy"], "is_contrarian", "wallet")
report("is_contrarian ~ recent_accuracy", beta, se, nc, n, ["recent_accuracy"])
print(f"test A done [{time.time()-t0:.1f}s]")

print("\n" + "=" * 70)
print("TEST B: exact-K persistence of contrarian tendency (wallet FE)")
print("=" * 70)
qualifying = qualifying.sort_values(["wallet", "seq"]).reset_index(drop=True)
for K in [1, 2, 5, 10, 25, 50, 100, 250, 500]:
    lag_col = f"lag{K}"
    qualifying[lag_col] = qualifying.groupby("wallet", sort=False)["is_contrarian"].shift(K)
    sub = qualifying.dropna(subset=[lag_col, "is_contrarian"])
    if sub["wallet"].nunique() < 5 or len(sub) < 50:
        print(f"K={K}: insufficient data (n={len(sub)})")
        continue
    d3 = demean(sub, [lag_col, "is_contrarian"])
    beta3, se3, nc3, n3 = cluster_robust_ols(d3, [lag_col], "is_contrarian", "wallet")
    report(f"K={K}", beta3, se3, nc3, n3, [lag_col])
    print(f"  [{time.time()-t0:.1f}s]")
