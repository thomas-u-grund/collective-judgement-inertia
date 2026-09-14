"""One compact additional robustness check requested in review round 3:
does the persistence effect survive controlling for more than just market
extremity -- specifically market liquidity, market age at the time of the
bet, and a linear calendar-time trend -- to rule out that the standing
behavioural state reflects selection into particular kinds of markets or
periods of platform activity, rather than a property of the person.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import sys
sys.path.insert(0, str(Path(__file__).parent))
from final_robustness import load_and_score, add_disjoint, qualifying, cluster_robust_ols, demean  # noqa: E402
from category_check import add_recent_contrarian  # noqa: E402
from config import DK_MANIFOLD  # noqa: E402


def load_contract_covariates():
    c = ds.dataset(str(DK_MANIFOLD / "contracts_dataset")).to_table(
        columns=["contract_id", "totalLiquidity", "createdTime"]
    ).to_pandas()
    return c


if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()
    dj = add_disjoint(df)
    dj = add_recent_contrarian(dj)
    q = qualifying(dj).dropna(subset=["recent_contrarian_rate"])
    print(f"qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users  [{time.time()-t0:.1f}s]")

    cov = load_contract_covariates()
    q = q.merge(cov, on="contract_id", how="left")
    q["prob_extremity"] = (q.probBefore - 0.5).abs()
    q["log_liquidity"] = np.log1p(q.totalLiquidity.clip(lower=0))
    q["market_age_days"] = (q.created_time - q.createdTime) / 86_400_000.0
    q["log_market_age"] = np.log1p(q.market_age_days.clip(lower=0))
    epoch0 = q.created_time.min()
    q["calendar_days"] = (q.created_time - epoch0) / 86_400_000.0

    xcols = ["recent_contrarian_rate", "prob_extremity", "log_liquidity", "log_market_age", "calendar_days"]
    qq = q.dropna(subset=xcols + ["is_contrarian"]).copy()
    print(f"with market covariates: n={len(qq):,}  users={qq.user_id.nunique():,}")

    dd = demean(qq, xcols + ["is_contrarian"])
    beta, se, nc = cluster_robust_ols(dd, xcols, "is_contrarian", "user_id")
    print(f"\nn={len(dd):,}  users={nc:,}")
    for name, b, s in zip(xcols, beta, se):
        lo, hi = b - 1.96 * s, b + 1.96 * s
        print(f"  {name:20s} beta={b:+.5f}  se={s:.5f}  t={b/s:+.2f}  95% CI [{lo:+.5f}, {hi:+.5f}]")

    print("\n(comparison) bivariate recent_contrarian_rate only, same sample:")
    dd2 = demean(qq, ["recent_contrarian_rate", "is_contrarian"])
    beta2, se2, _ = cluster_robust_ols(dd2, ["recent_contrarian_rate"], "is_contrarian", "user_id")
    b, s = beta2[0], se2[0]
    print(f"  recent_contrarian_rate  beta={b:+.5f}  se={s:.5f}  t={b/s:+.2f}  "
          f"95% CI [{b-1.96*s:+.5f}, {b+1.96*s:+.5f}]")

    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
