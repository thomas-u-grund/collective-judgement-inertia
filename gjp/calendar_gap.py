"""
Calendar-gap persistence check for GJP, matching main text Methods exactly:
is_contrarian_it = alpha_i + b1*C_it + b2*(C_it x log(1+gap_it)) + b3*log(1+gap_it),
C_it = recent contrarian rate (rolling last 20), gap_it = hours since the
person's PREVIOUS decision, person FE via demeaning, person-clustered SEs,
full covariance matrix for delta-method predicted-slope SEs.
"""
import numpy as np
import pandas as pd

decisions = pd.read_parquet("gjp_decisions_panel_matched.parquet")
decisions = decisions.sort_values(["user_id", "seq"]).reset_index(drop=True)

# recent contrarian rate (rolling last 20), matching persistence_matched.py
n = len(decisions)
recent_rate = np.full(n, np.nan)
gap_hours = np.full(n, np.nan)
for uid, idx in decisions.groupby("user_id", sort=False).indices.items():
    idx = np.asarray(idx)
    c = decisions["is_contrarian"].values[idx].astype(float)
    t = decisions["timestamp"].values[idx]
    m = len(c)
    cum = np.concatenate([[0.0], np.cumsum(c)])
    pos = np.arange(m)
    start = np.clip(pos - 20, 0, None)
    win_n = pos - start
    rate = np.where(win_n > 0, (cum[pos] - cum[start]) / np.maximum(win_n, 1), np.nan)
    recent_rate[idx] = rate
    t = pd.to_datetime(t)
    gaps = np.full(m, np.nan)
    gaps[1:] = (t[1:] - t[:-1]).total_seconds() / 3600.0
    gap_hours[idx] = gaps

decisions["recent_contrarian_rate"] = recent_rate
decisions["gap_hours"] = gap_hours
decisions["log_gap"] = np.log1p(decisions["gap_hours"])
decisions["interact"] = decisions["recent_contrarian_rate"] * decisions["log_gap"]

q = decisions.dropna(subset=["recent_contrarian_rate", "gap_hours", "is_contrarian"])
q = q[q["gap_hours"] > 0]
print(f"n={len(q):,} users={q.user_id.nunique():,}")

xcols = ["recent_contrarian_rate", "interact", "log_gap"]
means = q.groupby("user_id")[xcols + ["is_contrarian"]].transform("mean")
d = q.copy()
for c in xcols + ["is_contrarian"]:
    d[f"{c}_dm"] = d[c] - means[c]

X = np.column_stack([d[f"{c}_dm"].values for c in xcols] + [np.ones(len(d))])
y = d["is_contrarian_dm"].values
XtX_inv = np.linalg.inv(X.T @ X)
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta

codes = pd.factorize(d["user_id"].values)[0]
order = np.argsort(codes, kind="stable")
codes_sorted = codes[order]
Xs = X[order]
us = resid[order]
boundaries = np.flatnonzero(np.diff(codes_sorted)) + 1
starts = np.concatenate([[0], boundaries])
ends = np.concatenate([boundaries, [len(codes_sorted)]])
meat = np.zeros((X.shape[1], X.shape[1]))
for s, e in zip(starts, ends):
    Xg = Xs[s:e]
    ug = us[s:e]
    score_g = Xg.T @ ug
    meat += np.outer(score_g, score_g)
nc = len(starts)
nn, k = X.shape
correction = (nc / (nc - 1)) * ((nn - 1) / (nn - k))
cov = correction * XtX_inv @ meat @ XtX_inv

b1, b2, b3, b0 = beta
print(f"b1 (recent_contrarian_rate) = {b1:+.5f}")
print(f"b2 (interact)               = {b2:+.5f}")
print(f"b3 (log_gap)                = {b3:+.5f}")

cov_b1 = cov[0, 0]
cov_b2 = cov[1, 1]
cov_b1b2 = cov[0, 1]

for gap in [0.01, 24, 72, 168]:
    lg = np.log1p(gap)
    slope = b1 + b2 * lg
    var_slope = cov_b1 + (lg ** 2) * cov_b2 + 2 * lg * cov_b1b2
    se_slope = np.sqrt(var_slope)
    lo, hi = slope - 1.96 * se_slope, slope + 1.96 * se_slope
    print(f"gap={gap:>6.2f}h  predicted slope={slope:+.5f}  se={se_slope:.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")
