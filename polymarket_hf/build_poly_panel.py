"""
Build the Manifold/GJP-style decision panel from the filtered Polymarket
taker-BUY trades on resolved binary markets (poly_panel_raw.parquet).

is_contrarian: bought the token priced below 0.5 (buying the side the
  market currently prices as less likely) -- the same buy-only contrarian
  construction used for Manifold.
recent_accuracy: expanding accuracy over the wallet's own PAST trades
  whose market had ALREADY resolved (end_date) by this trade's timestamp.
  Computed via a vectorized per-wallet sort + searchsorted (equivalent to
  the heap-based sweep used for GJP, but avoids a 98M-row Python loop).
"""
import numpy as np
import pandas as pd

df = pd.read_parquet(
    "poly_panel_raw.parquet",
    columns=["wallet", "market_id", "timestamp", "end_date", "price", "correct"],
)
print("loaded:", df.shape)

df["timestamp"] = df["timestamp"].astype("int64")
end_date = pd.to_datetime(df["end_date"], errors="coerce")
assert str(end_date.dtype).startswith("datetime64[us"), end_date.dtype
df["end_date"] = end_date.astype("int64") // 10**6  # datetime64[us] -> seconds
df["price"] = df["price"].astype("float32")
df["correct"] = df["correct"].astype("int8")
df["is_contrarian"] = (df["price"] < 0.5).astype("float32")

df = df.sort_values(["wallet", "timestamp"]).reset_index(drop=True)
df["seq"] = df.groupby("wallet", sort=False).cumcount() + 1
print("sorted, seq assigned")

recent_acc = np.full(len(df), np.nan, dtype="float32")
n_resolved_so_far = np.zeros(len(df), dtype="int32")

for wallet, idx in df.groupby("wallet", sort=False).indices.items():
    idx = np.asarray(idx)
    t = df["timestamp"].values[idx]
    ed = df["end_date"].values[idx]
    c = df["correct"].values[idx].astype("float64")

    order_ed = np.argsort(ed, kind="stable")
    ed_sorted = ed[order_ed]
    c_sorted_cumsum = np.concatenate([[0.0], np.cumsum(c[order_ed])])

    n_before = np.searchsorted(ed_sorted, t, side="left")
    sum_before = c_sorted_cumsum[n_before]
    with np.errstate(invalid="ignore", divide="ignore"):
        ra = np.where(n_before > 0, sum_before / n_before, np.nan)
    recent_acc[idx] = ra
    n_resolved_so_far[idx] = n_before

df["recent_accuracy"] = recent_acc
df["n_resolved_so_far"] = n_resolved_so_far

print(df[["seq", "price", "is_contrarian", "recent_accuracy", "n_resolved_so_far"]].describe())
df.to_parquet("poly_decisions_panel.parquet", index=False)
print("saved panel:", df.shape, "wallets:", df["wallet"].nunique())
