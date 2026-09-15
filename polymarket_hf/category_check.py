"""
Category generalization test for Polymarket, matching
sim/manifold_prototype/category_check.py's logic exactly (same joint
model, same same-category/different-category next-bet test, same
within-category breakdown), but with keyword lists extended to match
Polymarket's actual content (crypto up/down markets by coin name,
sports by league abbreviation) rather than forcing Manifold's literal
keyword list onto different platform vocabulary.
"""
import time
import numpy as np
import pandas as pd

CATS = {
    "politics": ["politics", "election", "president", "congress", "senate", "democrat", "republican"],
    "sports": ["sports", "nba", "nfl", "nhl", "mlb", "soccer", "football", "baseball", "hockey",
               "tennis", "olympic", "atp", "wta", "itf", "ncaa", "fifa", "fif-", "ufc", "mma",
               "premier league", "champions league"],
    "tech_ai": ["ai-", "technology", "artificial-intelligence", "openai", "nvidia", "-ai", "llm", "gpt"],
    "crypto_finance": ["crypto", "stock", "finance", "bitcoin", "economics", "fed-", "ethereum",
                        "solana", "xrp", "dogecoin", "bnb", "up or down", "nasdaq", "s&p"],
    "science": ["science", "space", "health", "medicine", "climate", "temperature"],
}
PRIORITY = ["politics", "sports", "tech_ai", "crypto_finance", "science"]


def classify(text):
    s = str(text).lower()
    for cat in PRIORITY:
        for kw in CATS[cat]:
            if kw in s:
                return cat
    return None


def cluster_reg(d, xcols, ycol, id_col="wallet"):
    dd = d.dropna(subset=xcols + [ycol]).copy()
    if len(dd) < 500 or dd[id_col].nunique() < 20:
        return None
    for c in xcols + [ycol]:
        dd[c + "_dm"] = dd[c] - dd.groupby(id_col)[c].transform("mean")
    X = np.column_stack([dd[c + "_dm"] for c in xcols] + [np.ones(len(dd))])
    y = dd[ycol + "_dm"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    codes = pd.factorize(dd[id_col].values)[0]
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
    XtX_inv = np.linalg.inv(X.T @ X)
    nc = len(starts)
    n, k = X.shape
    corr = (nc / (nc - 1)) * ((n - 1) / (n - k))
    cov = corr * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, se, len(dd), nc


t0 = time.time()
markets = pd.read_parquet("markets.parquet", columns=["id", "question"])
markets["category"] = markets["question"].apply(classify)
cat_map = markets.set_index("id")["category"]
print("tagged markets:", cat_map.notna().sum(), "/", len(cat_map),
      f"({cat_map.notna().mean()*100:.1f}%)")
print(cat_map.value_counts())

skill = pd.read_parquet("poly_decisions_panel_matched.parquet")
raw = pd.read_parquet("poly_decisions_panel.parquet", columns=["wallet", "seq", "market_id"])
skill = skill.merge(raw, on=["wallet", "seq"], how="left")
skill["category"] = skill["market_id"].map(cat_map)
print(f"decisions with a category: {skill['category'].notna().sum():,} / {len(skill):,} [{time.time()-t0:.1f}s]")

q = skill.dropna(subset=["recent_skill", "baseline_skill", "recent_contrarian_rate"]) if "recent_contrarian_rate" in skill.columns else None
if q is None:
    # recompute recent_contrarian_rate (rolling last-20 is_contrarian), matching joint_horserace.py
    skill = skill.sort_values(["wallet", "seq"]).reset_index(drop=True)
    n = len(skill)
    recent_rate = np.full(n, np.nan)
    for wallet, idx in skill.groupby("wallet", sort=False).indices.items():
        idx = np.asarray(idx)
        c = skill["is_contrarian"].values[idx].astype(float)
        m = len(c)
        cum = np.concatenate([[0.0], np.cumsum(c)])
        pos = np.arange(m)
        start = np.clip(pos - 20, 0, None)
        win_n = pos - start
        rate = np.where(win_n > 0, (cum[pos] - cum[start]) / np.maximum(win_n, 1), np.nan)
        recent_rate[idx] = rate
    skill["recent_contrarian_rate"] = recent_rate
    print(f"recent_contrarian_rate built [{time.time()-t0:.1f}s]")

q = skill.dropna(subset=["recent_skill", "baseline_skill", "recent_contrarian_rate"])
xcols = ["recent_skill", "baseline_skill", "recent_contrarian_rate"]

print("\n" + "=" * 70)
print("(A) Same-category vs different-category NEXT decision")
print("=" * 70)


def next_info(d):
    d = d.sort_values("seq").reset_index(drop=True)
    d["next_is_contrarian"] = d["is_contrarian"].shift(-1)
    d["next_category"] = d["category"].shift(-1)
    d["same_category_next"] = d["next_category"] == d["category"]
    return d


cols = list(q.columns)
q_next = []
for _, d in q.groupby("wallet", sort=False):
    q_next.append(next_info(d))
q_next = pd.concat(q_next, ignore_index=True)
q_next = q_next.dropna(subset=["next_is_contrarian", "category", "next_category"])
print(f"[{time.time()-t0:.1f}s] next-decision panel: {len(q_next):,}")

for label, sub in [("same category", q_next[q_next["same_category_next"]]),
                    ("different category", q_next[~q_next["same_category_next"]])]:
    res = cluster_reg(sub, xcols, "next_is_contrarian")
    if res is None:
        print(f"  {label:20s} too few obs")
        continue
    beta, se, n, nc = res
    print(f"  {label:20s} n={n:9,d} wallets={nc:6,d}  recent_contrarian_rate: "
          f"beta={beta[2]:+.5f} se={se[2]:.5f} t={beta[2]/se[2]:+.2f}")

print("\n" + "=" * 70)
print("(B) Effect size within each category (contemporaneous)")
print("=" * 70)
for cat in PRIORITY:
    sub = q[q["category"] == cat]
    res = cluster_reg(sub, xcols, "is_contrarian")
    if res is None:
        print(f"  {cat:20s} too few obs (n={len(sub):,})")
        continue
    beta, se, n, nc = res
    print(f"  {cat:20s} n={n:9,d} wallets={nc:6,d}  recent_contrarian_rate: "
          f"beta={beta[2]:+.5f} se={se[2]:.5f} t={beta[2]/se[2]:+.2f}")
print(f"[{time.time()-t0:.1f}s]")
