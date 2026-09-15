"""
Build a Manifold-style decision panel from the Good Judgment Project (GJP)
survey-forecast data: one row per (user_id, ifp_id, forecast_id) "decision"
(new or updated probability forecast on a binary IFP), with:
  - value: the user's stated probability of answer option 'a'
  - prob_true: probability the user assigned to the outcome that actually resolved
  - consensus_loo: leave-one-out mean of all OTHER users' current standing
    forecast on this IFP at the moment of this decision (forward-filled)
  - departure: value - consensus_loo (signed distance from the crowd)
  - is_contrarian: 1 if the user's implied favored side differs from the
    crowd's implied favored side (analogous to the Manifold contrarian measure)
  - recent_accuracy: expanding mean of prob_true over the user's own PAST
    decisions whose IFP had already resolved by the time of this decision
    (no look-ahead; computed via a heap-based sweep per user)
  - seq: 1-indexed order of this decision within the user's own sequence
"""
import heapq
import pandas as pd
import numpy as np

BASE = "."

ifps = pd.read_csv(f"{BASE}/ifps.csv", encoding="latin-1")
ifps["n_opts"] = pd.to_numeric(ifps["n_opts"], errors="coerce")
binary_resolved = ifps[
    (ifps["q_type"] == 0) & (ifps["n_opts"] == 2) & (ifps["q_status"] == "closed") &
    (ifps["outcome"].isin(["a", "b"]))
].copy()
binary_resolved["date_closed"] = pd.to_datetime(binary_resolved["date_closed"], errors="coerce")
binary_resolved = binary_resolved.dropna(subset=["date_closed"])
print("binary resolved IFPs:", len(binary_resolved))

ifp_outcome = binary_resolved.set_index("ifp_id")["outcome"]
ifp_closed = binary_resolved.set_index("ifp_id")["date_closed"]

frames = []
for yr in [1, 2, 3, 4]:
    df = pd.read_csv(f"{BASE}/survey_fcasts.yr{yr}.tab", sep="\t")
    df = df[df["ifp_id"].isin(binary_resolved["ifp_id"])]
    df = df[df["answer_option"] == "a"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    frames.append(df[["ifp_id", "user_id", "forecast_id", "fcast_type", "value",
                       "timestamp", "ctt", "training"]])
    print(f"year {yr}: kept {len(df)} rows on binary-resolved IFPs")

all_df = pd.concat(frames, ignore_index=True)
all_df["outcome"] = all_df["ifp_id"].map(ifp_outcome)
all_df["date_closed"] = all_df["ifp_id"].map(ifp_closed)
all_df["prob_true"] = np.where(all_df["outcome"] == "a", all_df["value"], 1 - all_df["value"])

# --- leave-one-out running crowd consensus per IFP (forward-filled state) ---
all_df = all_df.sort_values(["ifp_id", "timestamp"]).reset_index(drop=True)
consensus = np.full(len(all_df), np.nan)

for ifp_id, grp in all_df.groupby("ifp_id", sort=False):
    idx = grp.index.to_numpy()
    users = grp["user_id"].to_numpy()
    values = grp["value"].to_numpy()
    last_value = {}
    total = 0.0
    n = 0
    for k in range(len(idx)):
        u = users[k]
        if u in last_value:
            prev = last_value[u]
            loo_total = total - prev
            loo_n = n - 1
        else:
            loo_total = total
            loo_n = n
        consensus[idx[k]] = (loo_total / loo_n) if loo_n > 0 else np.nan
        if u in last_value:
            total += values[k] - last_value[u]
        else:
            total += values[k]
            n += 1
        last_value[u] = values[k]

all_df["consensus_loo"] = consensus
all_df["departure"] = all_df["value"] - all_df["consensus_loo"]
all_df["is_contrarian"] = (
    (np.sign(all_df["value"] - 0.5) != np.sign(all_df["consensus_loo"] - 0.5))
    & (all_df["consensus_loo"] != 0.5)
).astype(float)
all_df.loc[all_df["consensus_loo"].isna(), "is_contrarian"] = np.nan

# --- restrict to genuine decisions (new / update), matching Manifold's buy-only filter ---
decisions = all_df[all_df["fcast_type"].isin([0, 1])].copy()
decisions = decisions.dropna(subset=["consensus_loo"])
decisions = decisions.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
decisions["seq"] = decisions.groupby("user_id").cumcount() + 1
print("decisions with defined consensus:", len(decisions))

# --- recent accuracy: expanding mean of prob_true on PAST, already-resolved IFPs only ---
recent_acc = np.full(len(decisions), np.nan)
n_resolved_so_far = np.full(len(decisions), 0, dtype=int)

for user_id, grp in decisions.groupby("user_id", sort=False):
    idx = grp.index.to_numpy()
    t = grp["timestamp"].to_numpy()
    closed = grp["date_closed"].to_numpy()
    pt = grp["prob_true"].to_numpy()
    order = np.argsort(t)
    idx, t, closed, pt = idx[order], t[order], closed[order], pt[order]

    pending = []  # heap of (closed_time, prob_true)
    resolved_sum = 0.0
    resolved_n = 0
    for k in range(len(idx)):
        cur_t = t[k]
        while pending and pending[0][0] < cur_t:
            _, p = heapq.heappop(pending)
            resolved_sum += p
            resolved_n += 1
        if resolved_n > 0:
            recent_acc[idx[k]] = resolved_sum / resolved_n
        n_resolved_so_far[idx[k]] = resolved_n
        heapq.heappush(pending, (closed[k], pt[k]))

decisions["recent_accuracy"] = recent_acc
decisions["n_resolved_so_far"] = n_resolved_so_far

decisions.to_parquet(f"{BASE}/gjp_decisions_panel.parquet", index=False)
print("saved panel:", decisions.shape)
print(decisions[["seq", "value", "consensus_loo", "departure", "is_contrarian",
                  "recent_accuracy", "n_resolved_so_far"]].describe())
