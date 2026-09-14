"""A true event-time plot around resolution feedback: for each of a user's
resolved bets (a 'feedback event'), track contrarian status at relative
decision positions -5..-1 (before the event becomes the most recent
feedback) and 0..+19 (the event's anchor decision and afterward), split by
whether that event resolved correct or incorrect.

Decisions -5..-1 serve as a placebo/pre-trend check: if feedback were
confounded with pre-existing behavioural differences, contrarian rate
would already differ there. Decisions 0..+19 show the actual response.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from final_robustness import load_and_score  # noqa: E402
from feedback_event_study import add_last_feedback  # noqa: E402

OFFSETS = list(range(-5, 20))  # -5..-1, 0..19


def per_user_events(d: pd.DataFrame):
    d = d.sort_values("created_time").reset_index(drop=True)
    n = len(d)
    is_c = d.is_contrarian.values.astype(float)
    r = d.n_known_prior_fb.values
    fb = d.last_feedback_correct.values

    # anchor position for each distinct r value >= 1: first row with that r
    change = np.r_[True, r[1:] != r[:-1]]
    anchor_rows = np.nonzero(change)[0]
    anchor_r = r[anchor_rows]
    anchor_fb = fb[anchor_rows]
    valid = anchor_r >= 1  # r=0 means no feedback yet, not a real event
    anchor_rows = anchor_rows[valid]
    anchor_fb = anchor_fb[valid]
    if len(anchor_rows) == 0:
        return None

    recs = []
    for off in OFFSETS:
        idx = anchor_rows + off
        ok = (idx >= 0) & (idx < n) & ~np.isnan(anchor_fb)
        if not ok.any():
            continue
        recs.append(pd.DataFrame({
            "offset": off,
            "is_contrarian": is_c[idx[ok]],
            "feedback_correct": anchor_fb[ok],
        }))
    if not recs:
        return None
    return pd.concat(recs, ignore_index=True)


if __name__ == "__main__":
    t0 = time.time()
    df = load_and_score()
    dj = add_last_feedback(df)
    print(f"scored + feedback-tagged: {len(dj):,} rows  [{time.time()-t0:.1f}s]")

    all_recs = []
    for uid, d in dj.groupby("user_id", sort=False):
        if len(d) < 60:
            continue
        rec = per_user_events(d)
        if rec is not None:
            all_recs.append(rec)
    panel = pd.concat(all_recs, ignore_index=True)
    print(f"event-time panel: {len(panel):,} (event, offset) observations  [{time.time()-t0:.1f}s]")

    summary = panel.groupby(["offset", "feedback_correct"]).is_contrarian.agg(["mean", "sem", "count"]).reset_index()
    print(summary.to_string(index=False))

    OUT = Path(__file__).resolve().parent.parent / "results" / "manifold_prototype" / "feedback_event_time.csv"
    summary.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}")
    print(f"total elapsed {time.time()-t0:.1f}s")
