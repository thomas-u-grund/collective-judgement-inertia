"""Corrected version: demeans recent_contrarian_rate and is_contrarian using
each person's mean computed over their FULL qualifying panel (not the
gap-restricted subsample), then restricts to gap-filtered rows for the
regression. This avoids the small-sample-per-user demeaning distortion of
the first version, where a user with only ~6 observations in the
gap>=168h subsample had their person-mean computed from just those 6
noisy observations.
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
from calendar_gap_persistence import add_gap, GAP_THRESHOLDS_HOURS  # noqa: E402


def demean_full_panel(full: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Demean using means computed over the FULL panel, returned as a
    lookup merged back in (so subsetting afterward doesn't change the
    demeaning basis)."""
    out = full.copy()
    for c in cols:
        out[f"{c}_dm"] = out[c] - out.groupby("user_id")[c].transform("mean")
    return out


def run_reg_fixed(dd, xcols, ycol, label):
    dd = dd.dropna(subset=[f"{c}_dm" for c in xcols] + [f"{ycol}_dm"]).copy()
    if len(dd) < 500 or dd.user_id.nunique() < 20:
        print(f"  {label:40s} too few obs (n={len(dd)})")
        return
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

    # Demean ONCE using the full panel's person-means.
    q_dm = demean_full_panel(q, ["recent_contrarian_rate", "is_contrarian"])

    print("=" * 78)
    print("FIXED: person-means computed from the FULL panel, not the gap-subsample")
    print("=" * 78)
    for hrs in GAP_THRESHOLDS_HOURS:
        sub = q_dm[q_dm.gap_hours >= hrs] if hrs > 0 else q_dm
        n_per_user = len(sub) / sub.user_id.nunique() if len(sub) else 0
        label = f"gap >= {hrs}h" if hrs > 0 else "all decisions (baseline)"
        print(f"  [{label}: avg {n_per_user:.1f} obs/user in this subsample]")
        run_reg_fixed(sub, ["recent_contrarian_rate"], "is_contrarian", label)
        print()

    print(f"total elapsed {time.time()-t0:.1f}s")
