"""Continuous version of the calendar-gap persistence check, replacing the
discrete-threshold-subsample approach (which required an awkward choice
between two different demeaning schemes -- within-subsample demeaning is
a valid but low-precision fixed-effects estimator on a restricted sample;
full-panel demeaning-then-restricting is a different estimand entirely,
not a "correction" of the first).

Instead, estimate one person-fixed-effects model on the FULL panel (so
there is only one, unambiguous demeaning scheme), with recent contrarian
rate interacted with log(1+gap_hours) -- the time since the user's
previous decision -- plus a main effect for log(1+gap_hours) itself:

    is_contrarian_it = alpha_i + b1*C_it + b2*(C_it * log1p(gap_it))
                       + b3*log1p(gap_it) + e_it

b2 tests directly whether the persistence slope weakens as the gap since
the previous decision grows. Predicted slopes at representative gaps
(same-session, 24h, 72h, 7 days) are reported for interpretability.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from final_robustness import load_and_score, add_disjoint, qualifying, cluster_robust_ols, demean  # noqa: E402
from category_check import add_recent_contrarian  # noqa: E402
from calendar_gap_persistence import add_gap  # noqa: E402

XCOLS = ["recent_contrarian_rate", "log_gap", "rc_x_loggap"]


if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()
    dj = add_disjoint(df)
    dj = add_recent_contrarian(dj)
    dj = add_gap(dj)
    q = qualifying(dj).dropna(subset=["recent_contrarian_rate", "gap_hours"]).copy()
    print(f"qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users  [{time.time()-t0:.1f}s]\n")

    q["log_gap"] = np.log1p(q.gap_hours)
    q["rc_x_loggap"] = q.recent_contrarian_rate * q.log_gap

    dd = demean(q, XCOLS + ["is_contrarian"])
    beta, se, nc = cluster_robust_ols(dd, XCOLS, "is_contrarian", "user_id")
    print(f"n={len(dd):,}  users={nc:,}\n")
    for name, b, s in zip(XCOLS, beta, se):
        lo, hi = b - 1.96 * s, b + 1.96 * s
        print(f"  {name:16s} beta={b:+.6f}  se={s:.6f}  t={b/s:+.2f}  95% CI [{lo:+.6f}, {hi:+.6f}]")

    print("\nPredicted persistence slope (recent_contrarian_rate coefficient) at representative gaps:")
    print("(slope = b_main + b_int * log1p(gap_hours); delta-method SE)")
    # covariance needed for delta method -- recompute via full cov matrix
    X = np.column_stack([dd[f"{c}_dm"] for c in XCOLS] + [np.ones(len(dd))])
    y = dd["is_contrarian_dm"].values
    resid = y - X @ np.concatenate([beta, [0.0]]) if False else None
    # recompute cov matrix directly (cluster_robust_ols only returns se); redo inline
    beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_full
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in dd.groupby("user_id").indices.items():
        Xg = X[idx]
        ug = resid[idx]
        s_ = Xg.T @ ug
        meat += np.outer(s_, s_)
    XtX_inv = np.linalg.inv(X.T @ X)
    n_, k_ = X.shape
    corr = (nc / (nc - 1)) * ((n_ - 1) / (n_ - k_))
    cov = corr * XtX_inv @ meat @ XtX_inv

    for label, gap_h in [("same-session (~2 min)", 2 / 60), ("24h", 24.0), ("72h", 72.0), ("168h (7d)", 168.0)]:
        lg = np.log1p(gap_h)
        slope = beta[0] + beta[2] * lg
        # delta method: Var(b0 + b2*lg) = Var(b0) + lg^2*Var(b2) + 2*lg*Cov(b0,b2)
        var_slope = cov[0, 0] + lg**2 * cov[2, 2] + 2 * lg * cov[0, 2]
        se_slope = np.sqrt(var_slope)
        lo, hi = slope - 1.96 * se_slope, slope + 1.96 * se_slope
        print(f"  gap={label:24s} log1p(gap)={lg:6.3f}  slope={slope:+.5f}  se={se_slope:.5f}  "
              f"95% CI [{lo:+.5f}, {hi:+.5f}]")

    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
