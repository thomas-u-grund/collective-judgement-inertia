"""Tests whether the contrarian-behavioral-persistence finding is really
topic clustering in disguise: (A) does persistence survive when the next
bet is in a DIFFERENT category from the current one, and (B) does the
effect size vary meaningfully across categories.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from final_robustness import load_and_score, add_disjoint, qualifying, RECENT_WINDOW  # noqa: E402
from config import DK_MANIFOLD  # noqa: E402

CATS = {
    "politics": ["politics", "election", "president", "congress", "senate", "democrat", "republican"],
    "sports": ["sports", "nba", "nfl", "soccer", "football", "baseball", "hockey", "tennis", "olympic"],
    "tech_ai": ["ai-", "technology", "artificial-intelligence", "openai", "nvidia", "-ai", "llm", "gpt"],
    "crypto_finance": ["crypto", "stock", "finance", "bitcoin", "economics", "fed-"],
    "science": ["science", "space", "health", "medicine", "climate"],
}
PRIORITY = ["politics", "sports", "tech_ai", "crypto_finance", "science"]


def classify(slug: str):
    s = slug.lower()
    for cat in PRIORITY:
        for kw in CATS[cat]:
            if kw in s:
                return cat
    return None


def contract_categories():
    gs = pd.read_csv(DK_MANIFOLD / "contracts_groupSlugs.csv")
    gs["cat"] = gs.group_slug.apply(classify)
    tagged = gs.dropna(subset=["cat"])
    # priority order per contract: keep first match by PRIORITY rank
    tagged["rank"] = tagged.cat.map({c: i for i, c in enumerate(PRIORITY)})
    primary = tagged.sort_values("rank").drop_duplicates("contract_id", keep="first")
    return primary.set_index("contract_id").cat


def add_recent_contrarian(df, X=RECENT_WINDOW):
    """Fraction of the user's most recent (up to X) decisions, in placement
    order, that were contrarian -- irrespective of resolution status.
    Identical construction to GJP's and Polymarket's recent-contrarian-rate
    builders (simple positional window, no resolution-time filtering):
    recent accuracy/baseline accuracy legitimately require known outcomes,
    but contrarian status is observable at placement time, so this measure
    should not be restricted to already-resolved decisions.
    """
    def per_user(d):
        d = d.sort_values("created_time")
        c = d.is_contrarian.values.astype(float)
        n = len(c)
        cum = np.concatenate([[0.0], np.cumsum(c)])
        pos = np.arange(n)
        start = np.clip(pos - X, 0, None)
        win_n = pos - start
        rc = np.where(win_n > 0, (cum[pos] - cum[start]) / np.maximum(win_n, 1), np.nan)
        out = d.copy()
        out["recent_contrarian_rate"] = rc
        return out
    cols = list(df.columns)
    return df.groupby("user_id", group_keys=False)[cols].apply(per_user)


def cluster_reg(d, xcols, ycol):
    dd = d.dropna(subset=xcols + [ycol]).copy()
    if len(dd) < 500 or dd.user_id.nunique() < 20:
        return None
    for c in xcols + [ycol]:
        dd[c + "_dm"] = dd[c] - dd.groupby("user_id")[c].transform("mean")
    X = np.column_stack([dd[c + "_dm"] for c in xcols] + [np.ones(len(dd))])
    y = dd[ycol + "_dm"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in dd.groupby("user_id").indices.items():
        Xg = X[idx]
        ug = resid[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    XtX_inv = np.linalg.inv(X.T @ X)
    nc = dd.user_id.nunique()
    n, k = X.shape
    corr = (nc / (nc - 1)) * ((n - 1) / (n - k))
    cov = corr * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, se, len(dd), nc


if __name__ == "__main__":
    df = load_and_score()
    cat_map = contract_categories()
    df["category"] = df.contract_id.map(cat_map)
    print(f"bets with a category: {df.category.notna().sum():,} / {len(df):,}")

    dj = add_disjoint(df)
    dj = add_recent_contrarian(dj)
    q = qualifying(dj).dropna(subset=["recent_contrarian_rate"])
    print(f"qualifying panel: {len(q):,} rows, {q.user_id.nunique():,} users\n")

    xcols = ["recent_skill", "baseline_skill", "recent_contrarian_rate"]

    print("=" * 70)
    print("(A) Same-category vs different-category NEXT bet")
    print("=" * 70)

    def next_bet_info(d: pd.DataFrame) -> pd.DataFrame:
        d = d.sort_values("created_time").reset_index(drop=True)
        d["next_is_contrarian"] = d.is_contrarian.shift(-1)
        d["next_category"] = d.category.shift(-1)
        d["same_category_next"] = d.next_category == d.category
        return d

    q2 = q.groupby("user_id", group_keys=False)[list(q.columns) + []].apply(
        lambda d: next_bet_info(d)) if False else None
    # simpler: merge next-bet info via groupby apply preserving all cols
    cols = list(q.columns)
    q_next = q.groupby("user_id", group_keys=False)[cols].apply(next_bet_info)
    q_next = q_next.dropna(subset=["next_is_contrarian", "category", "next_category"])

    for label, sub in [("same category", q_next[q_next.same_category_next]),
                        ("different category", q_next[~q_next.same_category_next])]:
        res = cluster_reg(sub, xcols, "next_is_contrarian")
        if res is None:
            print(f"  {label:20s} too few obs")
            continue
        beta, se, n, nc = res
        print(f"  {label:20s} n={n:8d} users={nc:5d}  recent_contrarian: "
              f"beta={beta[2]:+.5f} se={se[2]:.5f} t={beta[2]/se[2]:+.2f}   "
              f"recent_skill t={beta[0]/se[0]:+.2f}  baseline_skill t={beta[1]/se[1]:+.2f}")

    print("\n" + "=" * 70)
    print("(B) Effect size within each category (contemporaneous)")
    print("=" * 70)
    for cat in PRIORITY:
        sub = q[q.category == cat]
        res = cluster_reg(sub, xcols, "is_contrarian")
        if res is None:
            print(f"  {cat:16s} too few obs (n={len(sub)})")
            continue
        beta, se, n, nc = res
        print(f"  {cat:16s} n={n:8d} users={nc:5d}  recent_contrarian: "
              f"beta={beta[2]:+.5f} se={se[2]:.5f} t={beta[2]/se[2]:+.2f}   "
              f"recent_skill t={beta[0]/se[0]:+.2f}  baseline_skill t={beta[1]/se[1]:+.2f}")
