"""The definitive version of the first-bet-per-market check: predictor,
anchor, and target are ALL built exclusively from first bets on distinct
markets. recent_contrarian_rate is recomputed here as the rate over the
preceding 20 first-bets-on-distinct-markets only (not the full trading
history, as in first_bet_persistence.py), so repeated within-market
trading cannot contribute to the measured state at all.
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
from forward_horizon_exact import add_exact_k  # noqa: E402

K_LIST = [1, 20, 100]

if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()
    dj = add_disjoint(df)
    q = qualifying(dj)
    print(f"full qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users  [{time.time()-t0:.1f}s]")

    q_first = q.sort_values("created_time").drop_duplicates(subset=["user_id", "contract_id"], keep="first").copy()
    print(f"first-bet-per-market panel: {len(q_first):,} rows, {q_first.user_id.nunique():,} users")

    # recent_contrarian_rate computed EXCLUSIVELY within the first-bet-only
    # sequence -- the preceding 20 first-bets-on-distinct-markets, not the
    # full trading history.
    q_first = add_recent_contrarian(q_first)
    q_first = q_first.dropna(subset=["recent_contrarian_rate"]).copy()
    print(f"after distinct-market recent_contrarian_rate: {len(q_first):,} rows, "
          f"{q_first.user_id.nunique():,} users  [{time.time()-t0:.1f}s]")

    print("\n" + "=" * 78)
    print("Bivariate: distinct-market recent_contrarian_rate -> exact-K status,")
    print("predictor, anchor, and target ALL restricted to first-bets-on-distinct-markets")
    print("=" * 78)
    for K in K_LIST:
        qk = add_exact_k(q_first, K)
        dd = qk.dropna(subset=["recent_contrarian_rate", f"exact_{K}"]).copy()
        dd = demean(dd, ["recent_contrarian_rate", f"exact_{K}"])
        beta, se, nc = cluster_robust_ols(dd, ["recent_contrarian_rate"], f"exact_{K}", "user_id")
        b, s = beta[0], se[0]
        print(f"  K={K:4d}  n={len(dd):9,d}  users={nc:6,d}  beta={b:+.5f}  se={s:.5f}  "
              f"t={b/s:+.2f}  95% CI [{b-1.96*s:+.5f}, {b+1.96*s:+.5f}]")

    print("\n" + "=" * 78)
    print("Joint model (recent_skill, baseline_skill, distinct-market recent_contrarian_rate)")
    print("=" * 78)
    xcols = ["recent_skill", "baseline_skill", "recent_contrarian_rate"]
    for K in K_LIST:
        qk = add_exact_k(q_first, K)
        dd = qk.dropna(subset=xcols + [f"exact_{K}"]).copy()
        dd = demean(dd, xcols + [f"exact_{K}"])
        beta, se, nc = cluster_robust_ols(dd, xcols, f"exact_{K}", "user_id")
        print(f"  K={K:4d}  n={len(dd):9,d}  users={nc:6,d}")
        for name, b, s in zip(xcols, beta, se):
            print(f"      {name:22s} beta={b:+.5f}  se={s:.5f}  t={b/s:+.2f}  "
                  f"95% CI [{b-1.96*s:+.5f}, {b+1.96*s:+.5f}]")

    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
