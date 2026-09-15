"""
Continuous performance measure for GJP, replacing binary side-correctness
with prob_true (the probability the forecaster assigned to the outcome
that actually occurred -- a continuous performance score, not a Brier-loss
transform: 1-Brier = 2*prob_true - prob_true**2 for the binary case, a
distinct, monotonically related quantity) in the identical rolling-20/
expanding-baseline windowing as the main text and as matched_skill.py,
then rerun the joint model.
"""
import numpy as np
import pandas as pd

K_MIN_KNOWN = 10
MIN_BASELINE = 20
MIN_QUALIFYING_ROWS = 20
RECENT_WINDOW = 20

decisions = pd.read_parquet("gjp_decisions_panel.parquet")
decisions["score"] = decisions["prob_true"]  # continuous, 1 - Brier/2 equivalent scale


def add_disjoint(df, id_col, res_time_col, cur_time_col, score_col, X=RECENT_WINDOW):
    out = []
    for _, d in df.groupby(id_col, sort=False):
        by_res = d.sort_values(res_time_col)
        res_times = by_res[res_time_col].values
        score = by_res[score_col].values.astype(float)
        cum = np.concatenate([[0.0], np.cumsum(score)])
        ct = d[cur_time_col].values
        n_known = np.searchsorted(res_times, ct, side="right")
        window_start = np.clip(n_known - X, 0, None)
        recent_n = n_known - window_start
        recent_skill = np.where(recent_n > 0, (cum[n_known] - cum[window_start]) / np.maximum(recent_n, 1), np.nan)
        baseline_n = window_start
        baseline_skill = np.where(baseline_n > 0, cum[window_start] / np.maximum(baseline_n, 1), np.nan)
        d = d.copy()
        d["recent_skill_brier"] = recent_skill
        d["baseline_skill_brier"] = baseline_skill
        d["n_known_prior"] = n_known
        d["baseline_n"] = baseline_n
        out.append(d)
    return pd.concat(out, ignore_index=True)


decisions = add_disjoint(decisions, "user_id", "date_closed", "timestamp", "score")

q = decisions[(decisions["n_known_prior"] >= K_MIN_KNOWN) & (decisions["baseline_n"] >= MIN_BASELINE)].dropna(
    subset=["recent_skill_brier", "baseline_skill_brier", "is_contrarian"]
).copy()
counts = q.groupby("user_id").size()
keep = counts[counts >= MIN_QUALIFYING_ROWS].index
q = q[q["user_id"].isin(keep)].copy()
print(f"qualifying (Brier-based): n={len(q):,} users={q.user_id.nunique():,}")

# recent_contrarian_rate, matching joint model construction
q = q.sort_values(["user_id", "seq"]).reset_index(drop=True)
n = len(q)
recent_rate = np.full(n, np.nan)
for uid, idx in q.groupby("user_id", sort=False).indices.items():
    idx = np.asarray(idx)
    c = q["is_contrarian"].values[idx].astype(float)
    m = len(c)
    cum = np.concatenate([[0.0], np.cumsum(c)])
    pos = np.arange(m)
    start = np.clip(pos - RECENT_WINDOW, 0, None)
    win_n = pos - start
    rate = np.where(win_n > 0, (cum[pos] - cum[start]) / np.maximum(win_n, 1), np.nan)
    recent_rate[idx] = rate
q["recent_contrarian_rate"] = recent_rate
q = q.dropna(subset=["recent_contrarian_rate"])
print(f"final: n={len(q):,} users={q.user_id.nunique():,}")


def cluster_robust_ols(d, xcols, ycol, cluster_col):
    X = np.column_stack([d[f"{c}_dm"].values for c in xcols] + [np.ones(len(d))])
    y = d[f"{ycol}_dm"].values
    XtX_inv = np.linalg.inv(X.T @ X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    codes = pd.factorize(d[cluster_col].values)[0]
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    Xs = X[order]; us = resid[order]
    boundaries = np.flatnonzero(np.diff(codes_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(codes_sorted)]])
    meat = np.zeros((X.shape[1], X.shape[1]))
    for s, e in zip(starts, ends):
        Xg = Xs[s:e]; ug = us[s:e]
        sc = Xg.T @ ug
        meat += np.outer(sc, sc)
    n_clusters = len(starts)
    nn, k = X.shape
    correction = (n_clusters / (n_clusters - 1)) * ((nn - 1) / (nn - k))
    cov = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, se, n_clusters, nn


def demean(d, cols, id_col):
    d = d.copy()
    means = d.groupby(id_col)[cols].transform("mean")
    for c in cols:
        d[f"{c}_dm"] = d[c] - means[c]
    return d


xcols = ["recent_skill_brier", "baseline_skill_brier", "recent_contrarian_rate"]
d = demean(q, xcols + ["is_contrarian"], "user_id")
beta, se, nc, nn = cluster_robust_ols(d, xcols, "is_contrarian", "user_id")
print(f"\nGJP Brier-based joint model: n={nn:,} users={nc:,}")
for i, c in enumerate(xcols):
    b, s = beta[i], se[i]
    t = b / s
    print(f"  {c:>22s}: beta={b:+.5f}  se={s:.5f}  t={t:+.2f}  95% CI [{b-1.96*s:+.5f}, {b+1.96*s:+.5f}]")
