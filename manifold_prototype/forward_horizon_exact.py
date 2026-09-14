"""Responds to the reviewer point that the existing K=1..100 forward-horizon
check measures the AVERAGE contrarian rate over the next K decisions, not the
status of the decision exactly K steps ahead -- so persistence at K=100 was
never actually tested as such.

This script adds two more honest measures, both ordered by created_time
(decision order), on top of the same clean buy-only, disjoint-window panel
used everywhere else in the paper:

  (a) EXACT-K: is_contrarian status of the bet exactly K decisions later.
  (b) NON-OVERLAPPING WINDOWS: average is_contrarian over decisions
      [1-5], [6-10], [11-20], [21-50], [51-100] ahead (each window disjoint
      from the others, so a big window-51-100 coefficient means something
      cannot be produced by decisions 1-50 doing the work).

All regressions: within-person demeaned, person-clustered (sandwich) SEs,
recent_contrarian_rate as sole predictor (matching the horse-race winner).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from final_robustness import load_and_score, add_disjoint, qualifying, cluster_robust_ols, demean, RECENT_WINDOW  # noqa: E402
from category_check import add_recent_contrarian  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "manifold_prototype"

EXACT_K = [1, 5, 10, 20, 50, 100]
WINDOWS = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100)]


def add_exact_k(df: pd.DataFrame, K: int) -> pd.DataFrame:
    def per_user(d: pd.DataFrame) -> pd.DataFrame:
        d = d.sort_values("created_time")
        c = d.is_contrarian.values.astype(float)
        n = len(c)
        exact = np.full(n, np.nan)
        valid = np.arange(n) + K < n
        exact[valid] = c[np.arange(n)[valid] + K]
        out = d.copy()
        out[f"exact_{K}"] = exact
        return out
    cols = list(df.columns)
    return df.groupby("user_id", group_keys=False)[cols].apply(per_user)


def add_window(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Average is_contrarian over decisions [lo, hi] ahead, inclusive, 1-indexed."""
    def per_user(d: pd.DataFrame) -> pd.DataFrame:
        d = d.sort_values("created_time")
        c = d.is_contrarian.values.astype(float)
        n = len(c)
        cum = np.concatenate([[0.0], np.cumsum(c)])
        idx = np.arange(n)
        start = idx + lo       # first index in window (1-indexed offset -> idx+lo)
        end = idx + hi + 1     # exclusive upper bound
        end_clamped = np.minimum(end, n)
        n_have = end_clamped - start
        width = hi - lo + 1
        avg = np.full(n, np.nan)
        full = (start < n) & (n_have == width)
        avg[full] = (cum[end_clamped[full]] - cum[start[full]]) / width
        out = d.copy()
        out[f"win_{lo}_{hi}"] = avg
        return out
    cols = list(df.columns)
    return df.groupby("user_id", group_keys=False)[cols].apply(per_user)


def run_reg(d, ycol, label):
    dd = d.dropna(subset=["recent_contrarian_rate", ycol]).copy()
    if len(dd) < 500 or dd.user_id.nunique() < 20:
        print(f"  {label:14s} too few obs (n={len(dd)})")
        return
    dd = demean(dd, ["recent_contrarian_rate", ycol])
    beta, se, nc = cluster_robust_ols(dd, ["recent_contrarian_rate"], ycol, "user_id")
    b, s = beta[0], se[0]
    ci_lo, ci_hi = b - 1.96 * s, b + 1.96 * s
    r, _ = stats.pearsonr(dd["recent_contrarian_rate_dm"], dd[f"{ycol}_dm"])
    print(f"  {label:14s} n={len(dd):9,d} users={nc:6,d}  "
          f"beta={b:+.5f}  se={s:.5f}  t={b/s:+.2f}  95% CI [{ci_lo:+.5f}, {ci_hi:+.5f}]  r={r:+.4f}")


if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()
    dj = add_disjoint(df)
    dj = add_recent_contrarian(dj)
    q = qualifying(dj).dropna(subset=["recent_contrarian_rate"])
    print(f"qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users  [{time.time()-t0:.1f}s]\n")

    print("=" * 78)
    print("(a) EXACT-K: recent_contrarian_rate -> is_contrarian of the decision")
    print("    exactly K steps ahead (not an average over 1..K)")
    print("=" * 78)
    for K in EXACT_K:
        qk = add_exact_k(q, K)
        run_reg(qk, f"exact_{K}", f"K={K}")

    print("\n" + "=" * 78)
    print("(b) NON-OVERLAPPING WINDOWS: average is_contrarian over decisions")
    print("    [lo,hi] ahead -- each window uses disjoint future decisions")
    print("=" * 78)
    for lo, hi in WINDOWS:
        qw = add_window(q, lo, hi)
        run_reg(qw, f"win_{lo}_{hi}", f"[{lo}-{hi}]")

    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
