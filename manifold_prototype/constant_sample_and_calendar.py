"""Two checks:
(1) Constant-sample decay: recompute the exact-K correlation curve (Fig 1a)
    restricted to ONLY the rows/users that qualify at K=100, to test whether
    the 1->100 decay shape partly reflects the changing qualifying sample
    rather than genuine within-person decay.
(2) Calendar-time translation: for the K=100 qualifying sample, compute the
    elapsed calendar time between the focal decision and the decision
    exactly 100 steps later, to give "100 decisions" a tangible timescale.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from final_robustness import load_and_score, add_disjoint, qualifying, demean  # noqa: E402
from category_check import add_recent_contrarian  # noqa: E402
from forward_horizon_exact import add_exact_k, EXACT_K  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()
    dj = add_disjoint(df)
    dj = add_recent_contrarian(dj)
    q = qualifying(dj).dropna(subset=["recent_contrarian_rate"])
    print(f"full qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users  [{time.time()-t0:.1f}s]")

    # (1) Constant-sample decay: restrict to rows where exact_100 exists,
    # BEFORE computing exact_K for smaller K, so the same rows are used throughout.
    q100 = add_exact_k(q, 100).dropna(subset=["exact_100"]).copy()
    print(f"constant (K=100-qualifying) sample: {len(q100):,} rows, {q100.user_id.nunique():,} users")

    print("\n" + "=" * 78)
    print("Exact-K correlation, CONSTANT sample (same rows/users at every K)")
    print("=" * 78)
    for K in EXACT_K:
        qk = add_exact_k(q100, K)
        dd = qk.dropna(subset=["recent_contrarian_rate", f"exact_{K}"]).copy()
        dd = demean(dd, ["recent_contrarian_rate", f"exact_{K}"])
        r, p = stats.pearsonr(dd["recent_contrarian_rate_dm"], dd[f"exact_{K}_dm"])
        print(f"  K={K:4d}  n={len(dd):9,d}  users={dd.user_id.nunique():6,d}  r={r:+.4f}")

    # (2) Calendar-time translation for K=100
    print("\n" + "=" * 78)
    print("Calendar time elapsed between focal decision and decision 100 steps later")
    print("=" * 78)

    def elapsed_100(d: pd.DataFrame) -> pd.DataFrame:
        d = d.sort_values("created_time").reset_index(drop=True)
        ct = d.created_time.values
        n = len(ct)
        elapsed = np.full(n, np.nan)
        if n > 100:
            elapsed[:n - 100] = ct[100:] - ct[:n - 100]
        out = d.copy()
        out["elapsed_ms_100"] = elapsed
        return out

    cols = list(q.columns)
    qe = q.groupby("user_id", group_keys=False)[cols].apply(elapsed_100)
    qe = qe.dropna(subset=["elapsed_ms_100"])
    days = qe["elapsed_ms_100"] / 86_400_000.0
    print(f"n={len(qe):,} decisions with a valid 100-ahead pair, {qe.user_id.nunique():,} users")
    print(f"  median elapsed: {days.median():.1f} days")
    print(f"  IQR: [{days.quantile(0.25):.1f}, {days.quantile(0.75):.1f}] days")
    print(f"  10th-90th pct: [{days.quantile(0.10):.1f}, {days.quantile(0.90):.1f}] days")

    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
