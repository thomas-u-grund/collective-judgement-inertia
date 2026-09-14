"""Mirrors the empirical recent-contrarian-rate -> future-contrarian
methodology on the habit model's simulated data, to check whether a purely
outcome-blind sticky-state process reproduces the observed slow-decay
persistence profile (K=1..100)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from model_habit import Config, simulate  # noqa: E402

RECENT_WINDOW = 20


def add_recent_rate(df, X=RECENT_WINDOW):
    def per_agent(d):
        d = d.sort_values("round")
        c = d.is_contrarian.values.astype(float)
        n = len(c)
        cum = np.concatenate([[0.0], np.cumsum(c)])
        idx = np.arange(n)
        start = np.clip(idx - X, 0, None)
        win_n = idx - start
        rate = np.where(win_n > 0, (cum[idx] - cum[start]) / np.maximum(win_n, 1), np.nan)
        out = d.copy()
        out["recent_rate"] = rate
        return out
    cols = list(df.columns)
    return df.groupby("agent", group_keys=False)[cols].apply(per_agent)


def add_forward(df, K):
    def per_agent(d):
        d = d.sort_values("round")
        c = d.is_contrarian.values.astype(float)
        n = len(c)
        cum = np.concatenate([[0.0], np.cumsum(c)])
        idx = np.arange(n)
        upper = np.minimum(idx + 1 + K, n)
        n_fwd = upper - (idx + 1)
        avg = np.where(n_fwd == K, (cum[upper] - cum[idx + 1]) / np.maximum(n_fwd, 1), np.nan)
        out = d.copy()
        out[f"fwd_{K}"] = avg
        return out
    cols = list(df.columns)
    return df.groupby("agent", group_keys=False)[cols].apply(per_agent)


def add_exact_k(df, K):
    def per_agent(d):
        d = d.sort_values("round")
        c = d.is_contrarian.values.astype(float)
        n = len(c)
        exact = np.full(n, np.nan)
        valid = np.arange(n) + K < n
        exact[valid] = c[np.arange(n)[valid] + K]
        out = d.copy()
        out[f"exact_{K}"] = exact
        return out
    cols = list(df.columns)
    return df.groupby("agent", group_keys=False)[cols].apply(per_agent)


def within_agent(d, x, y):
    dd = d.dropna(subset=[x, y]).copy()
    dd[f"{x}_dm"] = dd[x] - dd.groupby("agent")[x].transform("mean")
    dd[f"{y}_dm"] = dd[y] - dd.groupby("agent")[y].transform("mean")
    r, p = stats.pearsonr(dd[f"{x}_dm"], dd[f"{y}_dm"])
    slope, *_ = stats.linregress(dd[f"{x}_dm"], dd[f"{y}_dm"])[:1] + (None,)
    return slope, r, len(dd)


if __name__ == "__main__":
    df = simulate(Config())
    df = add_recent_rate(df)
    q = df[df["round"] >= 100].dropna(subset=["recent_rate"])
    print(f"n={len(q):,}, agents={q.agent.nunique():,}")
    for K in [1, 5, 10, 20, 50, 100]:
        qk = add_forward(q, K)
        slope, r, n = within_agent(qk, "recent_rate", f"fwd_{K}")
        print(f"K={K:4d}  slope={slope:+.5f}  r={r:+.4f}  n={n}")
