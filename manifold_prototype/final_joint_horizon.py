"""Final, corrected joint horse-race: recent_skill, baseline_skill, and
recent_contrarian_rate jointly predicting is_contrarian, at K=1 (contemporaneous)
and at true exact-K and non-overlapping-window future horizons -- replacing
the earlier version that used an average-over-1..K forward measure.
Reports beta, person-clustered SE, t, and 95% CI throughout.
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
from forward_horizon_exact import add_exact_k, add_window, EXACT_K, WINDOWS  # noqa: E402

XCOLS = ["recent_skill", "baseline_skill", "recent_contrarian_rate"]


def run_joint(d, ycol, label):
    dd = d.dropna(subset=XCOLS + [ycol]).copy()
    dd = demean(dd, XCOLS + [ycol])
    beta, se, nc = cluster_robust_ols(dd, XCOLS, ycol, "user_id")
    print(f"{label:14s} n={len(dd):9,d} users={nc:6,d}")
    for name, b, s in zip(XCOLS, beta, se):
        ci_lo, ci_hi = b - 1.96 * s, b + 1.96 * s
        print(f"    {name:22s} beta={b:+.5f}  se={s:.5f}  t={b/s:+.2f}  95% CI [{ci_lo:+.5f}, {ci_hi:+.5f}]")
    return beta, se


if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()
    dj = add_disjoint(df)
    dj = add_recent_contrarian(dj)
    q = qualifying(dj).dropna(subset=["recent_contrarian_rate"])
    print(f"qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users  [{time.time()-t0:.1f}s]\n")

    print("=" * 78)
    print("Contemporaneous (K=0 / current decision)")
    print("=" * 78)
    run_joint(q, "is_contrarian", "current")

    print("\n" + "=" * 78)
    print("EXACT-K future horizons")
    print("=" * 78)
    for K in EXACT_K:
        qk = add_exact_k(q, K)
        run_joint(qk, f"exact_{K}", f"K={K}")

    print("\n" + "=" * 78)
    print("NON-OVERLAPPING WINDOW future horizons")
    print("=" * 78)
    for lo, hi in WINDOWS:
        qw = add_window(q, lo, hi)
        run_joint(qw, f"win_{lo}_{hi}", f"[{lo}-{hi}]")

    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
