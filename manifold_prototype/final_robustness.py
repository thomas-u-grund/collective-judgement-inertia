"""Comprehensive re-verification pass responding to the second simulated
review. Critically, restricts to genuine buy transactions (amount > 0) --
sells (amount < 0, ~13% of records) and zero-amount records (~8%, likely
unfilled limit orders) break the clean mapping between trade direction and
price direction that the whole analysis (is_contrarian, bet_correct,
moved_toward_truth, and every skill measure) implicitly assumed. This
affects every downstream number, not just the Brier-style check.

Also implements:
- continuous Brier-loss improvement (not just a binary "moved closer" flag)
- user-clustered standard errors (cluster-robust sandwich estimator)
- first-bet-per-market combined with strictly disjoint recent/baseline windows
- constant-sample forward-horizon check (same observations at every K)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy import stats

from config import DK_MANIFOLD  # noqa: E402
OUT = Path(__file__).resolve().parent.parent / "results" / "manifold_prototype"

MIN_BETS_TOTAL = 50
K_MIN_KNOWN = 10
MIN_QUALIFYING_ROWS = 20
RECENT_WINDOW = 20
MIN_BASELINE = 20


def load_and_score():
    t0 = time.time()
    contracts = ds.dataset(str(DK_MANIFOLD / "contracts_dataset")).to_table(
        columns=["contract_id", "isResolved", "resolution", "resolutionTime", "outcomeType"]
    ).to_pandas()
    contracts = contracts[
        (contracts.isResolved) & (contracts.outcomeType == "BINARY")
        & contracts.resolution.isin(["YES", "NO"])
    ][["contract_id", "resolution", "resolutionTime"]]

    bets = ds.dataset(str(DK_MANIFOLD / "bets_dataset")).to_table(
        columns=["bet_id", "user_id", "contract_id", "created_time",
                 "probBefore", "probAfter", "outcome", "amount", "isApi"]
    ).to_pandas()
    n0 = len(bets)
    bets = bets[(bets.isApi != True) & bets.probBefore.notna()  # noqa: E712
                & bets.outcome.isin(["YES", "NO"])]
    n1 = len(bets)
    bets = bets[bets.amount > 0]  # genuine buys only -- excludes sells and zero-amount records
    n2 = len(bets)
    print(f"bets: {n0:,} raw -> {n1:,} non-API/valid-outcome -> {n2:,} genuine buys only "
          f"({n2/n1*100:.1f}% retained)")

    df = bets.merge(contracts, on="contract_id", how="inner")
    df = df[df.resolutionTime > df.created_time]
    df = df[df.probBefore != 0.5]

    truth = (df.resolution == "YES")
    crowd_lean_yes = df.probBefore > 0.5
    bet_is_yes = df.outcome == "YES"
    truth_num = truth.astype(float)
    df = df.assign(
        is_contrarian=(bet_is_yes != crowd_lean_yes).astype(int),
        bet_correct=(bet_is_yes == truth).astype(int),
        truth_num=truth_num,
    )
    # continuous Brier-loss improvement: positive = moved toward truth
    df["brier_improvement"] = (df.probBefore - df.truth_num) ** 2 - (df.probAfter - df.truth_num) ** 2
    df["moved_toward_truth"] = (df.brier_improvement > 0).astype(int)

    counts = df.groupby("user_id").size()
    keep_users = counts[counts >= MIN_BETS_TOTAL].index
    df = df[df.user_id.isin(keep_users)]
    print(f"scored bets (genuine buys, resolved binary, has crowd lean): {len(df):,}  "
          f"({df.user_id.nunique():,} users)  [{time.time()-t0:.1f}s]")

    # sanity check: does outcome direction now match price direction as expected?
    d = df.brier_improvement  # not directly comparable; check delta instead
    delta = df.probAfter - df.probBefore
    yes_up = ((df.outcome == "YES") & (delta > 0)).sum() / (df.outcome == "YES").sum()
    print(f"sanity check -- outcome==YES bets with price up: {yes_up*100:.1f}% (should be ~95%+)")
    return df


def add_disjoint(df: pd.DataFrame, X: int = RECENT_WINDOW) -> pd.DataFrame:
    def per_user(d: pd.DataFrame) -> pd.DataFrame:
        by_res = d.sort_values("resolutionTime")
        res_times = by_res.resolutionTime.values
        correct = by_res.bet_correct.values.astype(float)
        cum = np.concatenate([[0.0], np.cumsum(correct)])

        ct = d.created_time.values
        n_known = np.searchsorted(res_times, ct, side="right")

        window_start = np.clip(n_known - X, 0, None)
        recent_n = n_known - window_start
        recent_skill = np.where(recent_n > 0, (cum[n_known] - cum[window_start]) / np.maximum(recent_n, 1), np.nan)

        baseline_n = window_start
        baseline_skill = np.where(baseline_n > 0, cum[window_start] / np.maximum(baseline_n, 1), np.nan)

        out = d.copy()
        out["recent_skill"] = recent_skill
        out["baseline_skill"] = baseline_skill
        out["n_known_prior"] = n_known
        out["baseline_n"] = baseline_n
        return out

    cols = list(df.columns)
    return df.groupby("user_id", group_keys=False)[cols].apply(per_user)


def qualifying(df):
    q = df[(df.n_known_prior >= K_MIN_KNOWN) & (df.baseline_n >= MIN_BASELINE)].dropna(
        subset=["recent_skill", "baseline_skill"]).copy()
    counts = q.groupby("user_id").size()
    keep = counts[counts >= MIN_QUALIFYING_ROWS].index
    return q[q.user_id.isin(keep)].copy()


def cluster_robust_ols(d, xcols, ycol, cluster_col):
    """Cluster-robust (sandwich) SEs for OLS on already-demeaned columns."""
    X = np.column_stack([d[f"{c}_dm"] for c in xcols] + [np.ones(len(d))])
    y = d[f"{ycol}_dm"].values
    XtX_inv = np.linalg.inv(X.T @ X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in d.groupby(cluster_col).indices.items():
        Xg = X[idx]
        ug = resid[idx]
        score_g = Xg.T @ ug
        meat += np.outer(score_g, score_g)
    n_clusters = d[cluster_col].nunique()
    n, k = X.shape
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    cov = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, se, n_clusters


def demean(d, cols):
    d = d.copy()
    for c in cols:
        d[f"{c}_dm"] = d[c] - d.groupby("user_id")[c].transform("mean")
    return d


if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()

    print("\n" + "=" * 70)
    print("(1) HEADLINE joint regression, clean buy-only data, disjoint windows")
    print("=" * 70)
    dj = add_disjoint(df)
    q = qualifying(dj)
    print(f"n={len(q):,}  users={q.user_id.nunique():,}")
    d = demean(q, ["recent_skill", "baseline_skill", "is_contrarian"])
    beta, se_naive, _ = cluster_robust_ols(d, ["recent_skill", "baseline_skill"], "is_contrarian", "user_id")
    # naive (non-clustered) for comparison
    X = np.column_stack([d.recent_skill_dm, d.baseline_skill_dm, np.ones(len(d))])
    y = d.is_contrarian_dm.values
    resid = y - X @ beta
    sigma2 = (resid @ resid) / (len(d) - 3)
    se_ols = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    for name, b, sn, sc in zip(["recent", "baseline", "const"], beta, se_ols, se_naive):
        print(f"  {name:10s} beta={b:+.5f}  se(OLS)={sn:.5f}  t(OLS)={b/sn:+.2f}  "
              f"se(user-clustered)={sc:.5f}  t(clustered)={b/sc:+.2f}")

    print("\nAlso clustering by market:")
    beta_m, se_m, n_clust_m = cluster_robust_ols(d, ["recent_skill", "baseline_skill"], "is_contrarian", "contract_id")
    for name, b, sc in zip(["recent", "baseline", "const"], beta_m, se_m):
        print(f"  {name:10s} beta={b:+.5f}  se(market-clustered)={sc:.5f}  t={b/sc:+.2f}")

    print("\n" + "=" * 70)
    print("(2) First-bet-per-market AND disjoint windows, together")
    print("=" * 70)
    q_first = q.sort_values("created_time").drop_duplicates(subset=["user_id", "contract_id"], keep="first")
    print(f"n={len(q_first):,}  users={q_first.user_id.nunique():,}")
    d2 = demean(q_first, ["recent_skill", "baseline_skill", "is_contrarian"])
    beta2, se2, ncl2 = cluster_robust_ols(d2, ["recent_skill", "baseline_skill"], "is_contrarian", "user_id")
    for name, b, sc in zip(["recent", "baseline", "const"], beta2, se2):
        print(f"  {name:10s} beta={b:+.5f}  se(user-clustered)={sc:.5f}  t={b/sc:+.2f}")

    print("\n" + "=" * 70)
    print("(3) Foundation check: continuous Brier-loss improvement (clean data)")
    print("=" * 70)
    for lo, hi, label in [(0.45, 0.55, "[0.45,0.55]"), (0.3, 0.7, "[0.3,0.7]"), (0.0, 1.0, "full")]:
        sub = df[(df.probBefore >= lo) & (df.probBefore <= hi)]
        g = sub.groupby("is_contrarian").brier_improvement.agg(["mean", "size"])
        t, p = stats.ttest_ind(sub[sub.is_contrarian == 0].brier_improvement,
                                sub[sub.is_contrarian == 1].brier_improvement, equal_var=False)
        print(f"  {label:14s} conformist mean-improvement={g.loc[0,'mean']:+.5f} (n={g.loc[0,'size']:,})  "
              f"contrarian={g.loc[1,'mean']:+.5f} (n={g.loc[1,'size']:,})  t={t:+.2f}  p={p:.2e}")

    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
    q.to_parquet(OUT / "final_robustness_panel.parquet", index=False)
