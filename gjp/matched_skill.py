"""
Rebuild recent/baseline accuracy for GJP using EXACTLY the same construction
as Manifold's add_disjoint() in sim/manifold_prototype/final_robustness.py:
recent_skill = rolling last-20 known-prior (resolved) fraction correct;
baseline_skill = expanding fraction correct over known-prior outcomes
strictly before that window (requires >=20); n_known_prior >= 10 required;
>=20 qualifying rows per person required.
"""
import numpy as np
import pandas as pd

K_MIN_KNOWN = 10
MIN_BASELINE = 20
MIN_QUALIFYING_ROWS = 20
RECENT_WINDOW = 20

decisions = pd.read_parquet("gjp_decisions_panel.parquet")
decisions["correct"] = (decisions["prob_true"] > 0.5).astype(float)  # side-correctness, matches Manifold's bet_correct


def add_disjoint(df, id_col, res_time_col, cur_time_col, correct_col, X=RECENT_WINDOW):
    out = []
    for _, d in df.groupby(id_col, sort=False):
        by_res = d.sort_values(res_time_col)
        res_times = by_res[res_time_col].values
        correct = by_res[correct_col].values.astype(float)
        cum = np.concatenate([[0.0], np.cumsum(correct)])

        ct = d[cur_time_col].values
        n_known = np.searchsorted(res_times, ct, side="right")

        window_start = np.clip(n_known - X, 0, None)
        recent_n = n_known - window_start
        recent_skill = np.where(recent_n > 0, (cum[n_known] - cum[window_start]) / np.maximum(recent_n, 1), np.nan)

        baseline_n = window_start
        baseline_skill = np.where(baseline_n > 0, cum[window_start] / np.maximum(baseline_n, 1), np.nan)

        d = d.copy()
        d["recent_skill"] = recent_skill
        d["baseline_skill"] = baseline_skill
        d["n_known_prior"] = n_known
        d["baseline_n"] = baseline_n
        out.append(d)
    return pd.concat(out, ignore_index=True)


decisions = add_disjoint(decisions, "user_id", "date_closed", "timestamp", "correct")

q = decisions[(decisions["n_known_prior"] >= K_MIN_KNOWN) & (decisions["baseline_n"] >= MIN_BASELINE)].dropna(
    subset=["recent_skill", "baseline_skill", "is_contrarian"]
).copy()
counts = q.groupby("user_id").size()
keep = counts[counts >= MIN_QUALIFYING_ROWS].index
q = q[q["user_id"].isin(keep)].copy()
print(f"qualifying (matched windowing): n={len(q):,} users={q.user_id.nunique():,}")

q.to_parquet("gjp_decisions_panel_matched.parquet", index=False)
print(q[["recent_skill", "baseline_skill", "n_known_prior", "baseline_n"]].describe())
