"""Tests whether the persistence effect survives a meaningful break in
platform activity, to distinguish a genuine behavioural state from a
"trading-session mode" that resets whenever the user stops and restarts.

For each user, we compute the elapsed time since their previous decision.
For decisions preceded by a gap of at least 24h / 72h / 7 days, we test
whether recent_contrarian_rate (computed from the trailing 20 decisions
before the gap) still predicts is_contrarian on this first-after-the-gap
decision. If the association survives large gaps, that is much harder to
explain as within-session momentum.
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

GAP_THRESHOLDS_HOURS = [0, 24, 72, 168]  # 0 = no restriction (baseline), 1, 3, 7 days


def add_gap(df: pd.DataFrame) -> pd.DataFrame:
    def per_user(d: pd.DataFrame) -> pd.DataFrame:
        d = d.sort_values("created_time")
        ct = d.created_time.values
        gap_ms = np.concatenate([[np.nan], np.diff(ct)])
        out = d.copy()
        out["gap_hours"] = gap_ms / 3_600_000.0
        return out
    cols = list(df.columns)
    return df.groupby("user_id", group_keys=False)[cols].apply(per_user)


def run_reg(dd, xcols, ycol, label):
    dd = dd.dropna(subset=xcols + [ycol]).copy()
    if len(dd) < 500 or dd.user_id.nunique() < 20:
        print(f"  {label:40s} too few obs (n={len(dd)})")
        return
    dd = demean(dd, xcols + [ycol])
    beta, se, nc = cluster_robust_ols(dd, xcols, ycol, "user_id")
    print(f"  {label:40s} n={len(dd):9,d} users={nc:6,d}")
    for name, b, s in zip(xcols, beta, se):
        ci_lo, ci_hi = b - 1.96 * s, b + 1.96 * s
        print(f"      {name:24s} beta={b:+.5f}  se={s:.5f}  t={b/s:+.2f}  95% CI [{ci_lo:+.5f}, {ci_hi:+.5f}]")


if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()
    dj = add_disjoint(df)
    dj = add_recent_contrarian(dj)
    dj = add_gap(dj)
    q = qualifying(dj).dropna(subset=["recent_contrarian_rate", "gap_hours"])
    print(f"qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users  [{time.time()-t0:.1f}s]\n")

    print("Distribution of inter-decision gaps (hours):")
    print(q.gap_hours.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    print()

    print("=" * 78)
    print("Does recent_contrarian_rate predict is_contrarian on the decision")
    print("immediately following a gap of at least X hours since the last bet?")
    print("=" * 78)
    for hrs in GAP_THRESHOLDS_HOURS:
        sub = q[q.gap_hours >= hrs] if hrs > 0 else q
        label = f"gap >= {hrs}h (n bets)" if hrs > 0 else "all decisions (baseline)"
        run_reg(sub, ["recent_contrarian_rate"], "is_contrarian", label)
        print()

    print(f"total elapsed {time.time()-t0:.1f}s")
