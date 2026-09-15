# Reliance on collective judgement shows behavioural inertia

Analysis and simulation code for the manuscript *"Reliance on collective
judgement shows behavioural inertia"* (submitted to *Nature Human
Behaviour*). This repository contains the scripts that produce every
number, table, and figure reported in the paper and its Supplementary
Information, across all three datasets analysed. It does not contain the
manuscript itself, the underlying data, or generated output files.

## Datasets

The paper analyses three independent, publicly available sources. This
repository does not redistribute any of them.

**Manifold Markets** (`manifold_prototype/`) -- bulk historical data
covering bets on resolved binary-outcome markets, 17 December 2021
through 4 July 2024. Obtain it from Manifold under their published data
terms (`docs.manifold.markets/data`) and place it in a local directory
containing (at minimum):

- `bets_dataset/` (Parquet)
- `contracts_dataset/` (Parquet)
- `contracts_groupSlugs.csv`

Point the scripts at that directory with an environment variable:

```bash
export MANIFOLD_DATA_DIR=/path/to/your/manifold/data
```

If unset, scripts default to `./data/manifold` relative to
`manifold_prototype/` (see `manifold_prototype/config.py`).

**Polymarket** (`polymarket_hf/`) -- trade-level data covering
Polymarket's full history, 2020--2026. Obtain it from
`huggingface.co/datasets/SII-WANGZJ/Polymarket_data` (MIT licence).
`extract_panel.py` queries the hosted Parquet files directly via DuckDB's
`httpfs` extension; downstream scripts expect the resulting intermediate
Parquet files (`poly_panel_raw.parquet`, `resolved_binary_markets.parquet`,
`wallet_counts.parquet`, etc.) in the working directory.

**Good Judgment Project** (`gjp/`) -- geopolitical-forecasting-tournament
survey data, 2011--2015. Obtain it from Harvard Dataverse
(`doi.org/10.7910/DVN/BPCDH5`); `build_panel.py` expects the raw
`ifps.csv` and `survey_fcasts.yr{1..4}.tab` files in the working
directory.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Polymarket's `extract_panel.py` additionally requires `duckdb`
(`pip install duckdb`), not listed in `requirements.txt` since it is only
needed for that one extraction step.

## Repository structure

- `manifold_prototype/` -- all empirical analyses on the Manifold bet
  data: sample construction, the main joint and persistence models, and
  every Manifold-specific robustness check reported in the paper and
  supplement.
- `gjp/` -- the equivalent pipeline on Good Judgment Project survey
  forecasts (Supplementary Note 1), reported as a structural check
  rather than an equivalent replication (its consensus was not visible
  to most participants, and its raw persistence result does not survive
  the distinct-question restriction -- see `core_tests.py` and
  `persistence_matched.py`).
- `polymarket_hf/` -- the equivalent pipeline on Polymarket trades
  (Supplementary Note 1), the paper's real-money replication dataset.
- `model_habit.py`, `analyze_habit.py`, `calibrate_habit_model.py` --
  the minimal outcome-blind AR(1) latent-state model and its calibration
  against the empirical Manifold decay curve.
- `make_figures.py` -- generates all main-text figures into a local
  `figures/` directory.

## A note on the recent-contrarian-rate construction

The single most consequential implementation detail in this codebase:
`recent_contrarian_rate` -- a person's/wallet's/forecaster's own
contrarian rate over their preceding 20 decisions, the dominant predictor
throughout the paper -- is built from **placement order, irrespective of
resolution status**, and **excludes the current decision**. This is
distinct from the two accuracy measures (`recent_skill`,
`baseline_skill`), which are necessarily restricted to decisions whose
outcome was already known at the time (accuracy cannot be scored on an
unresolved bet). An earlier version of `manifold_prototype/category_check.py`
mistakenly reused the resolution-time-filtered windowing built for the
accuracy measures when constructing `recent_contrarian_rate`, which
matters: on Manifold, a user's most recent 20 *resolved* decisions can
sit hundreds of decisions back in raw placement order, since many
markets take a long time to resolve. GJP's and Polymarket's scripts
(`gjp/matched_skill.py` and downstream; `polymarket_hf/matched_skill.py`
and downstream) always used the simple positional construction. The
current `manifold_prototype/category_check.py` (`add_recent_contrarian`)
matches that construction and has been independently audited against a
from-scratch reference implementation on several hundred randomly
sampled users, with zero unexplained discrepancies (all recovered
discrepancies traced to exact millisecond-timestamp ties in the raw
Manifold data).

## Running the pipeline

### Manifold (`manifold_prototype/`)

All scripts are runnable directly and print their results to stdout;
none require command-line arguments beyond `MANIFOLD_DATA_DIR` being
set. Two scripts write intermediate files that others depend on -- run
these first:

```bash
cd manifold_prototype
python final_robustness.py      # writes results/manifold_prototype/final_robustness_panel.parquet
python feedback_event_time.py   # writes results/manifold_prototype/feedback_event_time.csv (needed for the feedback event-time figure)
```

Then any of the following can be run independently:

| Script | Produces |
|---|---|
| `category_check.py` | Topic-generalisation figure and table; defines the canonical `add_recent_contrarian` |
| `forward_horizon_exact.py` | Exact-K persistence decay; non-overlapping-window robustness check |
| `final_joint_horizon.py` | Main joint model across horizons (K=0..100) |
| `decay_significance_test.py` | Pooled interaction test for the exact-K decay |
| `constant_sample_and_calendar.py` | Constant-sample robustness check; calendar-time-to-100-decisions statistic |
| `first_bet_persistence.py`, `first_bet_persistence_distinct.py` | First-bet-per-market and fully-distinct-market restrictions |
| `market_controls_robustness.py` | Market extremity, liquidity, age, calendar-trend controls |
| `feedback_event_study.py` | Resolution-feedback event study (main text and supplement) |
| `feedback_near_half.py` | Feedback robustness restricted to near-even-split resolutions |
| `brier_performance_check.py` | Brier-loss-improvement performance robustness check |
| `recent_window_sweep_corrected.py` | Recent-window-size sweep (5 to 640 decisions) |
| `calendar_gap_persistence.py`, `calendar_gap_persistence_v2.py`, `calendar_gap_continuous.py` | Calendar-gap persistence, discrete and continuous specifications |

Generate all figures with `python make_figures.py` from the repository
root.

### Good Judgment Project (`gjp/`)

Run `build_panel.py` first (reads `ifps.csv`,
`survey_fcasts.yr{1..4}.tab`; writes `gjp_decisions_panel.parquet`), then
`matched_skill.py` (writes `gjp_decisions_panel_matched.parquet`,
mirroring Manifold's accuracy-window construction). The remaining
scripts read those intermediate files directly: `joint_horserace.py`
(joint model), `persistence_matched.py` (exact-K persistence and the
distinct-question restriction), `brier_joint.py` (continuous
performance measure), `calendar_gap.py` (calendar-gap persistence),
`core_tests.py` (consensus-visibility subsample tests).

### Polymarket (`polymarket_hf/`)

Run `extract_panel.py` first (queries the hosted dataset via DuckDB;
writes `poly_panel_raw.parquet` and supporting Parquet files), then
`build_poly_panel.py` (writes the decision panel) and `matched_skill.py`
/ `matched_skill_full.py` (writes the near-resolution-excluded primary
panel and the full/uncleaned panel respectively, matching Manifold's
accuracy-window construction). The remaining scripts read those
intermediate files: `joint_horserace.py` / `joint_horserace_full.py`
(joint model, primary and full-sample specifications),
`persistence_matched.py` / `persistence_matched_final.py` (exact-K
persistence), `distinct_market.py` (distinct-market restriction),
`category_check.py` (topic generalisation), `human_plausible.py`
(high-frequency-wallet exclusion), `robustness_bot_check.py`
(near-resolution-threshold robustness), `threshold_sensitivity.py`
(full near-resolution-threshold sweep), `feedback_event_study.py`
(resolution-feedback event study), `brier_joint.py` (continuous
performance measure), `full_clean_ksweep.py` / `core_tests_poly.py`
(additional specification checks).

This reflects the analysis as actually run rather than a packaged
one-command pipeline; script docstrings state which construction and
which paper result each one produces.

## Software

Python 3.11+, with pandas, NumPy, PyArrow, SciPy, and Matplotlib (see
`requirements.txt` for tested versions); DuckDB for the Polymarket
extraction step only.

## License

Code released under the MIT License (see `LICENSE`).
