"""Reruns the headline recent/baseline performance regression using a
continuous Brier-loss-improvement measure of trading skill instead of
binary same-side correctness. Binary correctness can miscredit a
genuinely well-calibrated contrarian trade (e.g. buying NO against a 0.60
market when the true probability is 0.55, but YES ultimately resolves).
Brier-loss improvement -- (probBefore-truth)^2 - (probAfter-truth)^2 --
scores whether the trade moved the market closer to the eventual truth,
which is the more defensible measure of genuine forecasting skill.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from final_robustness import load_and_score, qualifying, cluster_robust_ols, demean, K_MIN_KNOWN, RECENT_WINDOW, MIN_BASELINE, MIN_QUALIFYING_ROWS  # noqa: E402
from category_check import add_recent_contrarian  # noqa: E402
from forward_horizon_exact import add_exact_k, EXACT_K  # noqa: E402

XCOLS = ["recent_skill_brier", "baseline_skill_brier", "recent_contrarian_rate"]


def add_disjoint_brier(df: pd.DataFrame, X: int = RECENT_WINDOW) -> pd.DataFrame:
    def per_user(d: pd.DataFrame) -> pd.DataFrame:
        by_res = d.sort_values("resolutionTime")
        res_times = by_res.resolutionTime.values
        brier = by_res.brier_improvement.values.astype(float)
        cum = np.concatenate([[0.0], np.cumsum(brier)])

        ct = d.created_time.values
        n_known = np.searchsorted(res_times, ct, side="right")

        window_start = np.clip(n_known - X, 0, None)
        recent_n = n_known - window_start
        recent_skill = np.where(recent_n > 0, (cum[n_known] - cum[window_start]) / np.maximum(recent_n, 1), np.nan)

        baseline_n = window_start
        baseline_skill = np.where(baseline_n > 0, cum[window_start] / np.maximum(baseline_n, 1), np.nan)

        out = d.copy()
        out["recent_skill_brier"] = recent_skill
        out["baseline_skill_brier"] = baseline_skill
        out["n_known_prior"] = n_known
        out["baseline_n"] = baseline_n
        return out

    cols = list(df.columns)
    return df.groupby("user_id", group_keys=False)[cols].apply(per_user)


if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()
    dj = add_disjoint_brier(df)
    dj = add_recent_contrarian(dj)
    q = dj[(dj.n_known_prior >= K_MIN_KNOWN) & (dj.baseline_n >= MIN_BASELINE)].dropna(
        subset=["recent_skill_brier", "baseline_skill_brier", "recent_contrarian_rate"]).copy()
    counts = q.groupby("user_id").size()
    keep = counts[counts >= MIN_QUALIFYING_ROWS].index
    q = q[q.user_id.isin(keep)].copy()
    print(f"qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users  [{time.time()-t0:.1f}s]")

    print("\n" + "=" * 78)
    print("Joint model with Brier-loss-improvement performance measures (contemporaneous)")
    print("=" * 78)
    dd = demean(q, XCOLS + ["is_contrarian"])
    beta, se, nc = cluster_robust_ols(dd, XCOLS, "is_contrarian", "user_id")
    print(f"n={len(dd):,}  users={nc:,}")
    for name, b, s in zip(XCOLS, beta, se):
        lo, hi = b - 1.96 * s, b + 1.96 * s
        print(f"  {name:24s} beta={b:+.6f}  se={s:.6f}  t={b/s:+.2f}  95% CI [{lo:+.6f}, {hi:+.6f}]")

    print("\n" + "=" * 78)
    print("Joint model with Brier-loss-improvement, at each exact-K horizon")
    print("=" * 78)
    for K in EXACT_K:
        qk = add_exact_k(q, K)
        dd = qk.dropna(subset=XCOLS + [f"exact_{K}"]).copy()
        dd = demean(dd, XCOLS + [f"exact_{K}"])
        beta, se, nc = cluster_robust_ols(dd, XCOLS, f"exact_{K}", "user_id")
        print(f"K={K:4d}  n={len(dd):9,d}  users={nc:6,d}")
        for name, b, s in zip(XCOLS, beta, se):
            lo, hi = b - 1.96 * s, b + 1.96 * s
            print(f"    {name:24s} beta={b:+.6f}  se={s:.6f}  t={b/s:+.2f}  95% CI [{lo:+.6f}, {hi:+.6f}]")

    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
