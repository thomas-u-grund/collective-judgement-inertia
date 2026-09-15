"""
Joint model matching Manifold's Table 2 / Fig 2 exactly:
is_contrarian ~ recent_skill + baseline_skill + recent_contrarian_rate,
person FE (within-person demeaning), person-clustered SEs.
"""
import numpy as np
import pandas as pd

RECENT_WINDOW = 20


def add_recent_rate(df, id_col, seq_col, y_col, X=RECENT_WINDOW):
    df = df.sort_values([id_col, seq_col])
    out = []
    for _, d in df.groupby(id_col, sort=False):
        c = d[y_col].values.astype(float)
        n = len(c)
        cum = np.concatenate([[0.0], np.cumsum(c)])
        idx = np.arange(n)
        start = np.clip(idx - X, 0, None)
        win_n = idx - start
        rate = np.where(win_n > 0, (cum[idx] - cum[start]) / np.maximum(win_n, 1), np.nan)
        d = d.copy()
        d["recent_contrarian_rate"] = rate
        out.append(d)
    return pd.concat(out, ignore_index=True)


def cluster_robust_ols(d, xcols, ycol, cluster_col):
    X = np.column_stack([d[f"{c}_dm"].values for c in xcols] + [np.ones(len(d))])
    y = d[f"{ycol}_dm"].values
    XtX_inv = np.linalg.inv(X.T @ X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    codes = pd.factorize(d[cluster_col].values)[0]
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    Xs = X[order]
    us = resid[order]
    boundaries = np.flatnonzero(np.diff(codes_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(codes_sorted)]])
    for s, e in zip(starts, ends):
        Xg = Xs[s:e]
        ug = us[s:e]
        score_g = Xg.T @ ug
        meat += np.outer(score_g, score_g)
    n_clusters = len(starts)
    n, k = X.shape
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    cov = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, se, n_clusters, n


def demean(d, cols, id_col):
    d = d.copy()
    means = d.groupby(id_col)[cols].transform("mean")
    for c in cols:
        d[f"{c}_dm"] = d[c] - means[c]
    return d


def report(label, beta, se, n_clusters, n, xcols):
    print(f"\n-- {label} --  n={n:,}  groups={n_clusters:,}")
    for i, c in enumerate(xcols):
        b, s = beta[i], se[i]
        t = b / s
        lo, hi = b - 1.96 * s, b + 1.96 * s
        print(f"  {c:>22s}: beta={b:+.5f}  se={s:.5f}  t={t:+.2f}  95% CI [{lo:+.5f}, {hi:+.5f}]")


skill = pd.read_parquet("gjp_decisions_panel_matched.parquet")
skill = add_recent_rate(skill, "user_id", "seq", "is_contrarian")
q = skill.dropna(subset=["recent_skill", "baseline_skill", "recent_contrarian_rate", "is_contrarian"])
print(f"final joint sample: n={len(q):,} users={q.user_id.nunique():,}")

d = demean(q, ["recent_skill", "baseline_skill", "recent_contrarian_rate", "is_contrarian"], "user_id")
beta, se, nc, n = cluster_robust_ols(
    d, ["recent_skill", "baseline_skill", "recent_contrarian_rate"], "is_contrarian", "user_id"
)
report("GJP joint model", beta, se, nc, n, ["recent_skill", "baseline_skill", "recent_contrarian_rate"])
