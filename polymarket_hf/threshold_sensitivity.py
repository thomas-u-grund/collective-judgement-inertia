"""
Sensitivity of the Polymarket performance-departure result to the choice
of near-resolution exclusion threshold. The main text's primary
specification excludes trades placed within 24 hours of their market's
resolution; this reruns the identical joint model (recent accuracy,
baseline accuracy, recent contrarian rate jointly predicting current
contrarian status) at 1 minute, 1 hour, 6 hours, 24 hours, and 48 hours,
to show the 24-hour choice is not doing idiosyncratic work.
"""
import time
import numpy as np
import pandas as pd

K_MIN_KNOWN = 10
MIN_BASELINE = 20
MIN_QUALIFYING_ROWS = 20
RECENT_WINDOW = 20

THRESHOLDS = [
    ("1 minute", 60),
    ("1 hour", 3600),
    ("6 hours", 21600),
    ("24 hours", 86400),
    ("48 hours", 172800),
]


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


def demean(d, cols, id_col="wallet"):
    d = d.copy()
    means = d.groupby(id_col)[cols].transform("mean")
    for c in cols:
        d[f"{c}_dm"] = d[c] - means[c]
    return d


t0 = time.time()
raw = pd.read_parquet(
    "poly_decisions_panel.parquet",
    columns=["wallet", "seq", "timestamp", "end_date", "correct", "is_contrarian", "n_resolved_so_far"],
)
raw["gap_to_resolution"] = raw["end_date"] - raw["timestamp"]
print(f"loaded: {raw.shape} [{time.time()-t0:.1f}s]")

results = []
for label, gap_s in THRESHOLDS:
    df = raw[raw["gap_to_resolution"] > gap_s].copy()

    n = len(df)
    recent_skill = np.full(n, np.nan)
    baseline_skill = np.full(n, np.nan)
    n_known_prior = np.zeros(n, dtype="int32")
    baseline_n = np.zeros(n, dtype="int32")

    for wallet, idx in df.groupby("wallet", sort=False).indices.items():
        idx = np.asarray(idx)
        sub_t = df["timestamp"].values[idx]
        sub_end = df["end_date"].values[idx]
        sub_c = df["correct"].values[idx].astype(float)
        order_res = np.argsort(sub_end, kind="stable")
        res_sorted = sub_end[order_res]
        c_sorted = sub_c[order_res]
        cum = np.concatenate([[0.0], np.cumsum(c_sorted)])
        n_known = np.searchsorted(res_sorted, sub_t, side="right")
        window_start = np.clip(n_known - RECENT_WINDOW, 0, None)
        recent_n = n_known - window_start
        rs = np.where(recent_n > 0, (cum[n_known] - cum[window_start]) / np.maximum(recent_n, 1), np.nan)
        bn = window_start
        bs = np.where(bn > 0, cum[window_start] / np.maximum(bn, 1), np.nan)
        recent_skill[idx] = rs
        baseline_skill[idx] = bs
        n_known_prior[idx] = n_known
        baseline_n[idx] = bn

    df["recent_skill"] = recent_skill
    df["baseline_skill"] = baseline_skill
    df["n_known_prior"] = n_known_prior
    df["baseline_n"] = baseline_n

    q = df[(df["n_known_prior"] >= K_MIN_KNOWN) & (df["baseline_n"] >= MIN_BASELINE)].dropna(
        subset=["recent_skill", "baseline_skill", "is_contrarian"]
    ).copy()
    counts = q.groupby("wallet").size()
    keep = counts[counts >= MIN_QUALIFYING_ROWS].index
    q = q[q["wallet"].isin(keep)].copy()

    q = q.sort_values(["wallet", "seq"]).reset_index(drop=True)
    nn = len(q)
    recent_rate = np.full(nn, np.nan)
    for wallet, idx in q.groupby("wallet", sort=False).indices.items():
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

    d = demean(q, ["recent_skill", "baseline_skill", "recent_contrarian_rate", "is_contrarian"])
    beta, se, nc, n_final = cluster_robust_ols(
        d, ["recent_skill", "baseline_skill", "recent_contrarian_rate"], "is_contrarian", "wallet"
    )
    row = {
        "threshold": label, "gap_s": gap_s, "n": n_final, "wallets": nc,
        "recent_beta": beta[0], "recent_se": se[0],
        "baseline_beta": beta[1], "baseline_se": se[1],
        "rate_beta": beta[2], "rate_se": se[2],
    }
    results.append(row)
    print(f"{label:>10s} (>{gap_s:>6d}s): n={n_final:,} wallets={nc:,}  "
          f"recent_acc={beta[0]:+.4f} (t={beta[0]/se[0]:+.2f})  "
          f"baseline_acc={beta[1]:+.4f} (t={beta[1]/se[1]:+.2f})  "
          f"recent_rate={beta[2]:+.4f} (t={beta[2]/se[2]:+.2f})  [{time.time()-t0:.1f}s]")

pd.DataFrame(results).to_csv("threshold_sensitivity_results.csv", index=False)
print(f"\ntotal elapsed {time.time()-t0:.1f}s")
