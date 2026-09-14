# Reliance on collective judgement shows behavioural inertia

Analysis and simulation code for the manuscript *"Reliance on collective
judgement shows behavioural inertia"* (prepared for *Nature Human
Behaviour*). This repository contains the scripts that produce every
number, table, and figure reported in the paper and its Supplementary
Information. It does not contain the manuscript itself, the underlying
Manifold Markets data, or generated output files.

## Data

The analysis uses bulk historical data from
[Manifold Markets](https://manifold.markets), covering bets on resolved
binary-outcome markets from 17 December 2021 through 4 July 2024. This
repository does not redistribute that data. Obtain it from Manifold under
their published data terms, and place it in a local directory containing
(at minimum):

- `bets_dataset/` (Parquet)
- `contracts_dataset/` (Parquet)
- `contracts_groupSlugs.csv`

Point the scripts at that directory with an environment variable:

```bash
export MANIFOLD_DATA_DIR=/path/to/your/manifold/data
```

If unset, scripts default to looking for `./data/manifold` relative to
this repository (see `manifold_prototype/config.py`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Structure

- `manifold_prototype/` -- all empirical analyses on the Manifold bet
  data (sample construction, main regressions, every robustness check
  reported in the paper and supplement).
- `model_habit.py`, `analyze_habit.py`, `calibrate_habit_model.py` --
  the minimal outcome-blind AR(1) latent-state model (Supplementary
  Note 9) and its calibration against the empirical decay curve.
- `make_figures.py` -- generates all figures (main text Figs. 1-3 and
  Supplementary Figs. 2-3) into a local `figures/` directory.

## Running the pipeline

All `manifold_prototype/` scripts are runnable directly and print their
results to stdout; none require command-line arguments beyond
`MANIFOLD_DATA_DIR` being set. Each docstring states which paper result
it produces. Two scripts write intermediate files that others depend on
-- run these first:

```bash
cd manifold_prototype
python final_robustness.py      # writes results/manifold_prototype/final_robustness_panel.parquet
python feedback_event_time.py   # writes results/manifold_prototype/feedback_event_time.csv (needed for Fig. 5 / Supp. Fig. 3)
```

Then any of the following can be run independently:

| Script | Produces |
|---|---|
| `category_check.py` | Fig. 3 (topic generalisation); Table 4 |
| `forward_horizon_exact.py` | Fig. 1a/1b; Tables 2-3 (exact-K and disjoint-window methodology) |
| `final_joint_horizon.py` | Table 2 (joint model across horizons) |
| `decay_significance_test.py` | Pooled interaction test for the exact-K decay |
| `constant_sample_and_calendar.py` | Supplementary Note 8 (constant sample); calendar-time-to-100-decisions statistic |
| `first_bet_persistence.py`, `first_bet_persistence_distinct.py` | Supplementary Note 5 (first-bet-per-market, distinct-market predictor) |
| `market_controls_robustness.py` | Supplementary Note 4 (market extremity, liquidity, age, calendar trend) |
| `feedback_event_study.py` | Table 5; main text feedback results |
| `feedback_near_half.py` | Supplementary Note 6 (near-even-split feedback robustness) |
| `brier_performance_check.py` | Supplementary Note 7 (Brier-loss performance measure) |
| `recent_window_sweep_corrected.py` | Supplementary Note 3 (recent-window-size sweep) |
| `calendar_gap_persistence.py`, `calendar_gap_persistence_v2.py`, `calendar_gap_continuous.py` | Supplementary Note 10 (calendar-gap persistence, discrete and continuous specifications) |

From the repository root, generate all figures with:

```bash
python make_figures.py
```

## Software

Python 3.11+, with pandas, NumPy, PyArrow, SciPy, and Matplotlib (see
`requirements.txt` for tested versions).

## License

Code released under the MIT License (see `LICENSE`).
