"""Event study: does resolution feedback (the most recently resolved bet
coming in correct vs. incorrect) change a person's propensity to depart from
the crowd on their NEXT decision, controlling for the behavioral state they
were already in (recent_contrarian_rate)?

This directly targets the mediator-vs-confound ambiguity around
recent_contrarian_rate: if outcome feedback moves behavior net of prior
state, the "outcome-blind persistent state" story is wrong or incomplete. If
feedback has no detectable effect while prior state strongly predicts the
next decision, that is much more direct evidence for outcome-insensitivity
than comparing trailing averages.
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


def add_last_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """For each decision, find the outcome (correct/incorrect) of the most
    recently resolved bet strictly before this decision's created_time, and
    how long ago (in decisions) that feedback event was."""
    def per_user(d: pd.DataFrame) -> pd.DataFrame:
        by_res = d.sort_values("resolutionTime")
        res_times = by_res.resolutionTime.values
        correct = by_res.bet_correct.values.astype(float)
        res_order_idx = np.arange(len(by_res))

        d = d.sort_values("created_time")
        ct = d.created_time.values
        n_known = np.searchsorted(res_times, ct, side="right")
        last_idx = n_known - 1
        has_feedback = last_idx >= 0
        last_feedback_correct = np.where(has_feedback, correct[np.clip(last_idx, 0, None)], np.nan)

        out = d.copy()
        out["last_feedback_correct"] = last_feedback_correct
        out["n_known_prior_fb"] = n_known
        return out
    cols = list(df.columns)
    return df.groupby("user_id", group_keys=False)[cols].apply(per_user)


def run_reg(dd, xcols, ycol, label):
    dd = dd.dropna(subset=xcols + [ycol]).copy()
    if len(dd) < 500 or dd.user_id.nunique() < 20:
        print(f"  {label:28s} too few obs (n={len(dd)})")
        return
    dd = demean(dd, xcols + [ycol])
    beta, se, nc = cluster_robust_ols(dd, xcols, ycol, "user_id")
    X = np.column_stack([dd[f"{c}_dm"] for c in xcols] + [np.ones(len(dd))])
    y = dd[f"{ycol}_dm"].values
    resid = y - X @ beta
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    print(f"  {label:28s} n={len(dd):9,d} users={nc:6,d}  within-person R^2={r2:.5f}")
    for name, b, s in zip(xcols, beta, se):
        ci_lo, ci_hi = b - 1.96 * s, b + 1.96 * s
        print(f"      {name:24s} beta={b:+.5f}  se={s:.5f}  t={b/s:+.2f}  95% CI [{ci_lo:+.5f}, {ci_hi:+.5f}]")
    return r2


if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()
    dj = add_disjoint(df)
    dj = add_recent_contrarian(dj)
    dj = add_last_feedback(dj)
    q = qualifying(dj).dropna(subset=["recent_contrarian_rate", "last_feedback_correct"])
    print(f"qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users  [{time.time()-t0:.1f}s]\n")

    print("=" * 78)
    print("(1) Does the most recent feedback (last bet correct y/n) predict")
    print("    is_contrarian on the CURRENT decision, net of recent_contrarian_rate?")
    print("=" * 78)
    run_reg(q, ["last_feedback_correct", "recent_contrarian_rate"], "is_contrarian", "joint model")

    print("\n" + "=" * 78)
    print("(2) Bivariate check: feedback alone (no state control)")
    print("=" * 78)
    r2_fb_only = run_reg(q, ["last_feedback_correct"], "is_contrarian", "feedback only")
    r2_state_only = run_reg(q, ["recent_contrarian_rate"], "is_contrarian", "state only")
    r2_joint = run_reg(q, ["last_feedback_correct", "recent_contrarian_rate"], "is_contrarian", "joint (repeat)")
    if r2_fb_only is not None and r2_state_only is not None and r2_joint is not None:
        print(f"\n  Does feedback improve prediction beyond persistent state?")
        print(f"    R^2(state only)          = {r2_state_only:.5f}")
        print(f"    R^2(feedback only)       = {r2_fb_only:.5f}")
        print(f"    R^2(joint)               = {r2_joint:.5f}")
        print(f"    incremental R^2 from adding feedback to state = {r2_joint - r2_state_only:.5f}")

    print("\n" + "=" * 78)
    print("(3) Restrict to decisions immediately following feedback (same-day-ish):")
    print("    only the FIRST decision after each resolved bet, to avoid diluting")
    print("    a short-lived reaction across many intervening decisions")
    print("=" * 78)
    # first decision at each n_known_prior_fb value per user = first decision
    # observed right after that many bets have resolved
    q_first_after = q.sort_values(["user_id", "created_time"]).drop_duplicates(
        subset=["user_id", "n_known_prior_fb"], keep="first")
    print(f"  n={len(q_first_after):,} (vs {len(q):,} full panel)")
    run_reg(q_first_after, ["last_feedback_correct", "recent_contrarian_rate"], "is_contrarian",
            "joint, first-after-feedback")

    print("\n" + "=" * 78)
    print("(4) Does feedback affect the recent_contrarian_rate control itself?")
    print("    (sanity check on the mediator-vs-confound worry: if feedback")
    print("    predicts recent_contrarian_rate, it may already be baked in)")
    print("=" * 78)
    run_reg(q, ["last_feedback_correct"], "recent_contrarian_rate", "feedback -> state")

    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
