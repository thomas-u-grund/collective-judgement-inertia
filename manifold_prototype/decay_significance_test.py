"""Tests whether the exact-K persistence coefficient's decline with K is
statistically distinguishable from flat, rather than just eyeballing
overlapping CIs across separate per-K regressions. Pools the six exact-K
strata (K=1,5,10,20,50,100), each person-demeaned within its own qualifying
subsample (as in the per-K regressions), and estimates a single interaction
model: exact_K ~ recent_contrarian_rate + recent_contrarian_rate x log(K),
with person-clustered SEs. The interaction coefficient's t-stat is the
formal test of whether the slope changes with K.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from final_robustness import load_and_score, add_disjoint, qualifying, cluster_robust_ols  # noqa: E402
from category_check import add_recent_contrarian  # noqa: E402
from forward_horizon_exact import add_exact_k, EXACT_K  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()
    dj = add_disjoint(df)
    dj = add_recent_contrarian(dj)
    q = qualifying(dj).dropna(subset=["recent_contrarian_rate"])
    print(f"qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users  [{time.time()-t0:.1f}s]")

    stacks = []
    for K in EXACT_K:
        qk = add_exact_k(q, K)
        dd = qk.dropna(subset=["recent_contrarian_rate", f"exact_{K}"]).copy()
        dd["rc_dm"] = dd["recent_contrarian_rate"] - dd.groupby("user_id")["recent_contrarian_rate"].transform("mean")
        dd["y_dm"] = dd[f"exact_{K}"] - dd.groupby("user_id")[f"exact_{K}"].transform("mean")
        dd["logK"] = np.log(K)
        stacks.append(dd[["user_id", "rc_dm", "y_dm", "logK"]])
        print(f"  K={K:4d}  n={len(dd):9,d}  users={dd.user_id.nunique():6,d}")

    stacked = pd.concat(stacks, ignore_index=True)
    logK_mean = stacked["logK"].mean()
    stacked["logK_c"] = stacked["logK"] - logK_mean
    stacked["rc_x_logK"] = stacked["rc_dm"] * stacked["logK_c"]

    print(f"\nstacked n={len(stacked):,} rows across {len(EXACT_K)} horizons, "
          f"{stacked.user_id.nunique():,} unique users  [{time.time()-t0:.1f}s]")

    X = np.column_stack([stacked["rc_dm"].values, stacked["rc_x_logK"].values, np.ones(len(stacked))])
    y = stacked["y_dm"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in stacked.groupby("user_id").indices.items():
        Xg = X[idx]
        ug = resid[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    XtX_inv = np.linalg.inv(X.T @ X)
    nc = stacked.user_id.nunique()
    n, k = X.shape
    corr = (nc / (nc - 1)) * ((n - 1) / (n - k))
    cov = corr * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))

    names = ["recent_contrarian_rate (at mean logK)", "recent_contrarian_rate x logK (centred)", "const"]
    print(f"\nn={n:,}  users={nc:,}")
    for name, b, s in zip(names, beta, se):
        lo, hi = b - 1.96 * s, b + 1.96 * s
        print(f"  {name:42s} beta={b:+.5f}  se={s:.5f}  t={b/s:+.2f}  95% CI [{lo:+.5f}, {hi:+.5f}]")

    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
