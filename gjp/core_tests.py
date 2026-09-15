"""
Core replication tests on the GJP decision panel, using the exact same
cluster-robust (sandwich) within-person-demeaned OLS methodology as the
Manifold analysis (final_robustness.py: demean() + cluster_robust_ols()).

Test A (performance -> departure): does a forecaster's own recent accuracy
  on already-resolved IFPs predict whether their CURRENT forecast departs
  from the crowd (is_contrarian), within person?
Test B (persistence): does a forecaster's own contrarian tendency at
  decision t-K predict their tendency at decision t, within person, for a
  range of exact lags K?
"""
import numpy as np
import pandas as pd

BASE = "."


def cluster_robust_ols(d, xcols, ycol, cluster_col):
    X = np.column_stack([d[f"{c}_dm"].values for c in xcols] + [np.ones(len(d))])
    y = d[f"{ycol}_dm"].values
    XtX_inv = np.linalg.inv(X.T @ X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in d.groupby(cluster_col).indices.items():
        Xg = X[idx]
        ug = resid[idx]
        score_g = Xg.T @ ug
        meat += np.outer(score_g, score_g)
    n_clusters = d[cluster_col].nunique()
    n, k = X.shape
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    cov = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, se, n_clusters, n


def demean(d, cols):
    d = d.copy()
    for c in cols:
        d[f"{c}_dm"] = d[c] - d.groupby("user_id")[c].transform("mean")
    return d


def report(label, beta, se, n_clusters, n, xcols):
    print(f"\n-- {label} --  n={n:,}  users={n_clusters:,}")
    for i, c in enumerate(xcols):
        b, s = beta[i], se[i]
        t = b / s
        lo, hi = b - 1.96 * s, b + 1.96 * s
        print(f"  {c:>18s}: beta={b:+.5f}  se={s:.5f}  t={t:+.2f}  95% CI [{lo:+.5f}, {hi:+.5f}]")


decisions = pd.read_parquet(f"{BASE}/gjp_decisions_panel.parquet")
print("loaded panel:", decisions.shape, "users:", decisions.user_id.nunique())

# require at least 2 resolved IFPs behind them so recent_accuracy is meaningful
qualifying = decisions[decisions["n_resolved_so_far"] >= 2].dropna(
    subset=["recent_accuracy", "is_contrarian", "departure"]
).copy()
print("qualifying decisions:", len(qualifying), "users:", qualifying.user_id.nunique())

print("\n" + "=" * 70)
print("TEST A: recent accuracy -> departure from consensus (person FE)")
print("=" * 70)
d = demean(qualifying, ["recent_accuracy", "is_contrarian"])
beta, se, nc, n = cluster_robust_ols(d, ["recent_accuracy"], "is_contrarian", "user_id")
report("is_contrarian ~ recent_accuracy", beta, se, nc, n, ["recent_accuracy"])

d2 = demean(qualifying, ["recent_accuracy", "departure"])
beta2, se2, nc2, n2 = cluster_robust_ols(d2, ["recent_accuracy"], "departure", "user_id")
report("departure ~ recent_accuracy", beta2, se2, nc2, n2, ["recent_accuracy"])

print("\n" + "=" * 70)
print("TEST B: exact-K persistence of contrarian tendency (person FE)")
print("=" * 70)
qualifying = qualifying.sort_values(["user_id", "seq"]).reset_index(drop=True)
for K in [1, 2, 5, 10, 25, 50, 100]:
    lag_col = f"lag{K}"
    qualifying[lag_col] = qualifying.groupby("user_id")["is_contrarian"].shift(K)
    sub = qualifying.dropna(subset=[lag_col, "is_contrarian"]).copy()
    if sub.user_id.nunique() < 5 or len(sub) < 50:
        print(f"K={K}: insufficient data (n={len(sub)})")
        continue
    d3 = demean(sub, [lag_col, "is_contrarian"])
    beta3, se3, nc3, n3 = cluster_robust_ols(d3, [lag_col], "is_contrarian", "user_id")
    report(f"K={K}", beta3, se3, nc3, n3, [lag_col])
