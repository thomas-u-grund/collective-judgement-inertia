"""Corrected version of recent_window_sweep.py: the original used
reviewer_robustness.load_and_score, which predates the buy-only (amount>0)
correction and used naive (non-clustered) standard errors. This version
uses the same clean buy-only panel and person-clustered inference as every
other analysis in the final manuscript, so the supplement is consistent
with the main text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from final_robustness import load_and_score, cluster_robust_ols, demean, K_MIN_KNOWN  # noqa: E402

WINDOWS = [5, 10, 20, 40, 80, 160, 320, 640]
MIN_BASELINE = 20
MIN_QUALIFYING_ROWS = 20


def add_disjoint_x(df: pd.DataFrame, X: int) -> pd.DataFrame:
    def per_user(d: pd.DataFrame) -> pd.DataFrame:
        by_res = d.sort_values("resolutionTime")
        res_times = by_res.resolutionTime.values
        correct = by_res.bet_correct.values.astype(float)
        contrarian = by_res.is_contrarian.values.astype(float)
        cum = np.concatenate([[0.0], np.cumsum(correct)])
        cum_c = np.concatenate([[0.0], np.cumsum(contrarian)])
        ct = d.created_time.values
        n_known = np.searchsorted(res_times, ct, side="right")
        window_start = np.clip(n_known - X, 0, None)
        recent_n = n_known - window_start
        recent_skill = np.where(recent_n > 0, (cum[n_known] - cum[window_start]) / np.maximum(recent_n, 1), np.nan)
        recent_contrarian_rate = np.where(
            recent_n > 0, (cum_c[n_known] - cum_c[window_start]) / np.maximum(recent_n, 1), np.nan)
        baseline_n = window_start
        baseline_skill = np.where(baseline_n > 0, cum[window_start] / np.maximum(baseline_n, 1), np.nan)
        out = d.copy()
        out["recent_skill"] = recent_skill
        out["baseline_skill"] = baseline_skill
        out["recent_contrarian_rate"] = recent_contrarian_rate
        out["n_known_prior"] = n_known
        out["baseline_n"] = baseline_n
        return out
    cols = list(df.columns)
    return df.groupby("user_id", group_keys=False)[cols].apply(per_user)


XCOLS = ["recent_skill", "baseline_skill", "recent_contrarian_rate"]

if __name__ == "__main__":
    df = load_and_score()
    rows = []
    print("Two-covariate model (recent_skill, baseline_skill only -- omits recent_contrarian_rate):")
    print(f"{'X':>6s}  {'n_obs':>10s}  {'n_users':>8s}  {'recent_b':>9s}  {'recent_t':>8s}  "
          f"{'baseline_b':>10s}  {'baseline_t':>10s}")
    for X in WINDOWS:
        d = add_disjoint_x(df, X)
        q = d[(d.n_known_prior >= K_MIN_KNOWN) & (d.baseline_n >= MIN_BASELINE)].dropna(
            subset=["recent_skill", "baseline_skill"]).copy()
        counts = q.groupby("user_id").size()
        keep = counts[counts >= MIN_QUALIFYING_ROWS].index
        q = q[q.user_id.isin(keep)]
        if len(q) < 1000 or q.user_id.nunique() < 30:
            print(f"{X:6d}  too few observations (n={len(q)})")
            continue
        dd = demean(q, ["recent_skill", "baseline_skill", "is_contrarian"])
        beta2, se2, _ = cluster_robust_ols(dd, ["recent_skill", "baseline_skill"], "is_contrarian", "user_id")
        print(f"{X:6d}  {len(q):10d}  {q.user_id.nunique():8d}  {beta2[0]:+9.5f}  {beta2[0]/se2[0]:+8.2f}  "
              f"{beta2[1]:+10.5f}  {beta2[1]/se2[1]:+10.2f}")

    print("\nThree-covariate model (adds recent_contrarian_rate, matching main-text Table 2 methodology):")
    print(f"{'X':>6s}  {'n_obs':>10s}  {'n_users':>8s}  {'recent_b':>9s}  {'recent_t':>8s}  "
          f"{'baseline_b':>10s}  {'baseline_t':>10s}  {'contrarian_b':>13s}  {'contrarian_t':>13s}")
    for X in WINDOWS:
        d = add_disjoint_x(df, X)
        q = d[(d.n_known_prior >= K_MIN_KNOWN) & (d.baseline_n >= MIN_BASELINE)].dropna(
            subset=XCOLS).copy()
        counts = q.groupby("user_id").size()
        keep = counts[counts >= MIN_QUALIFYING_ROWS].index
        q = q[q.user_id.isin(keep)]
        if len(q) < 1000 or q.user_id.nunique() < 30:
            print(f"{X:6d}  too few observations (n={len(q)})")
            continue
        dd = demean(q, XCOLS + ["is_contrarian"])
        beta3, se3, nc = cluster_robust_ols(dd, XCOLS, "is_contrarian", "user_id")
        rows.append({"X": X, "n": len(q), "n_users": nc,
                      "recent_beta": beta3[0], "recent_se": se3[0],
                      "baseline_beta": beta3[1], "baseline_se": se3[1],
                      "contrarian_beta": beta3[2], "contrarian_se": se3[2]})
        print(f"{X:6d}  {len(q):10d}  {nc:8d}  {beta3[0]:+9.5f}  {beta3[0]/se3[0]:+8.2f}  "
              f"{beta3[1]:+10.5f}  {beta3[1]/se3[1]:+10.2f}  {beta3[2]:+13.5f}  {beta3[2]/se3[2]:+13.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).resolve().parent.parent / "results" / "manifold_prototype" / "recent_window_sweep_corrected.csv", index=False)
    print("\nsaved -> results/manifold_prototype/recent_window_sweep_corrected.csv")
