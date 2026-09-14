"""Restores and extends the first-bet-per-user-market robustness check for
the persistence result specifically (not just the original headline skill
regression). Concern: 'recently contrarian -> subsequently contrarian'
could partly reflect sticking to the same position on the same market
repeatedly, rather than a general behavioural state.

Fix: restrict to each user's FIRST bet on each market (dropping every
repeat bet on the same question), and recompute the exact-K forward
measure entirely within this reduced sequence -- so 'K decisions ahead'
means K distinct markets ahead, with no repeat-market bets anywhere in
either the anchor or the target. recent_skill/baseline_skill/
recent_contrarian_rate covariates are still computed on each user's full
history (as in the original first-bet check), since the behavioural state
itself should reflect everything the person actually did; only the rows
used to ANCHOR and TARGET the persistence estimate are restricted to
first-touches of a market.
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
    dj = add_recent_contrarian(dj)
    q = qualifying(dj).dropna(subset=["recent_contrarian_rate"])
    print(f"full qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users  [{time.time()-t0:.1f}s]")

    q_first = q.sort_values("created_time").drop_duplicates(subset=["user_id", "contract_id"], keep="first").copy()
    print(f"first-bet-per-market panel: {len(q_first):,} rows, {q_first.user_id.nunique():,} users "
          f"({len(q_first)/len(q)*100:.1f}% of full panel retained)")

    print("\n" + "=" * 78)
    print("Bivariate: recent_contrarian_rate -> exact-K status, first-bet-per-market only")
    print("(K now counted in distinct-market steps, since repeat-market bets are removed)")
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
    print("Joint model (recent_skill, baseline_skill, recent_contrarian_rate), first-bet-per-market only")
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
