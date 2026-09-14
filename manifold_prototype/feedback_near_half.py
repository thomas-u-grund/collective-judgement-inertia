"""Robustness check on the feedback event study: restricts to decisions
where the RESOLVING bet's own probBefore was close to 0.5 (genuinely
uncertain at the time it was placed). This weakens the mechanical link
between a person's prior contrarian tendency and whether that bet turned
out correct -- near 0.5, being contrarian confers little or no systematic
edge or handicap, unlike in the full sample where contrarian bets are
mechanically much less likely to be correct (34.1% vs 81.8%).
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
from feedback_event_study import run_reg  # noqa: E402


def add_last_feedback_with_prob(df: pd.DataFrame) -> pd.DataFrame:
    def per_user(d: pd.DataFrame) -> pd.DataFrame:
        by_res = d.sort_values("resolutionTime")
        res_times = by_res.resolutionTime.values
        correct = by_res.bet_correct.values.astype(float)
        prob_before = by_res.probBefore.values.astype(float)

        d = d.sort_values("created_time")
        ct = d.created_time.values
        n_known = np.searchsorted(res_times, ct, side="right")
        last_idx = n_known - 1
        has_feedback = last_idx >= 0
        last_feedback_correct = np.where(has_feedback, correct[np.clip(last_idx, 0, None)], np.nan)
        last_feedback_probbefore = np.where(has_feedback, prob_before[np.clip(last_idx, 0, None)], np.nan)

        out = d.copy()
        out["last_feedback_correct"] = last_feedback_correct
        out["last_feedback_probbefore"] = last_feedback_probbefore
        out["n_known_prior_fb"] = n_known
        return out
    cols = list(df.columns)
    return df.groupby("user_id", group_keys=False)[cols].apply(per_user)


if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()
    dj = add_disjoint(df)
    dj = add_recent_contrarian(dj)
    dj = add_last_feedback_with_prob(dj)
    q = qualifying(dj).dropna(subset=["recent_contrarian_rate", "last_feedback_correct", "last_feedback_probbefore"])
    print(f"qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users  [{time.time()-t0:.1f}s]\n")

    extremity = (q.last_feedback_probbefore - 0.5).abs()

    for lo, hi, label in [(0.0, 0.10, "resolving bet near 0.5: |p-0.5|<=0.10"),
                           (0.0, 0.05, "resolving bet near 0.5: |p-0.5|<=0.05"),
                           (0.10, 0.50, "resolving bet lopsided: |p-0.5|>0.10 (comparison)")]:
        sub = q[(extremity > lo) & (extremity <= hi)] if lo > 0 else q[extremity <= hi]
        print("=" * 78)
        print(f"{label}   (n={len(sub):,})")
        print("=" * 78)
        run_reg(sub, ["last_feedback_correct", "recent_contrarian_rate"], "is_contrarian", label[:28])
        print()

    print(f"total elapsed {time.time()-t0:.1f}s")
