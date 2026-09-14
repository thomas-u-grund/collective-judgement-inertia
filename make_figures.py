"""Figures for the reframed paper: persistence of contrarian behavior,
ruling out accuracy, ruling out topic-clustering, and the habit model.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model_habit import Config, simulate  # noqa: E402
from analyze_habit import add_recent_rate, add_exact_k, within_agent  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

BLACK = "#222222"
GRAY = "#333333"
MIDGRAY = "#8a8a8a"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.labelcolor": "#222222",
    "xtick.color": "#555555", "ytick.color": "#555555", "figure.dpi": 300,
})


def fig1_persistence():
    # Two panels: (a) exact-K correlation -- the contrarian status of the
    # decision exactly K steps ahead, not an average over decisions 1..K
    # (that earlier measure mechanically inflated apparent persistence at
    # large K by averaging away single-decision noise); (b) the joint-model
    # recent-contrarian-rate coefficient in five mutually disjoint future
    # windows (Table 3), showing the exact-K decay is not an artefact of a
    # single-decision measure -- a genuinely non-overlapping window as
    # distant as 51-100 decisions ahead still carries a detectable effect.
    K = [1, 5, 10, 20, 50, 100]
    r_data = [0.0319, 0.0301, 0.0280, 0.0265, 0.0224, 0.0199]

    win_labels = ["1-5", "6-10", "11-20", "21-50", "51-100"]
    win_x = [0, 1, 2, 3, 4]
    win_beta = [0.075, 0.070, 0.067, 0.062, 0.054]
    win_se = [0.012, 0.012, 0.012, 0.013, 0.013]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))

    ax = axes[0]
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.plot(K, r_data, marker="o", color=GRAY, lw=1.8, ms=5)
    ax.set_xscale("log")
    ax.set_xticks(K)
    ax.set_xticklabels([str(k) for k in K])
    ax.set_xlabel("Decisions ahead ($K$)")
    ax.set_ylabel("Correlation ($r$)")
    ax.set_title("(a) Exact decision $K$ steps ahead", fontsize=9)

    ax = axes[1]
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    cis = [1.96 * s for s in win_se]
    ypos = np.arange(len(win_labels))[::-1]
    ax.errorbar(win_beta, ypos, xerr=cis, fmt="o", color=BLACK, ecolor=BLACK,
                elinewidth=1.3, capsize=3, ms=5, mec="white", mew=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(win_labels, fontsize=8)
    ax.set_ylim(-0.7, len(win_labels) - 0.3)
    ax.set_ylabel("Non-overlapping future window\n(decisions ahead)")
    ax.set_xlabel("Coefficient, 95% CI")
    ax.set_title("(b) Disjoint future windows", fontsize=9)

    fig.tight_layout(pad=1.2)
    fig.savefig(FIGDIR / "fig1_persistence.pdf")
    plt.close(fig)


def fig2_horserace():
    # Coefficients (not t-statistics) with 95% CIs, from the joint within-
    # person model at K=1 (person-clustered SEs), as a forest plot: a point
    # estimate with a CI whisker communicates magnitude and uncertainty
    # directly, without the implicit area-encoding of a bar chart (which
    # can look misleading when a CI straddles zero, as two of these three do).
    labels = ["Recent accuracy", "Baseline accuracy", "Recent contrarian rate"]
    betas = [-0.0123, 0.0795, 0.0764]
    ses = [0.00743, 0.05168, 0.01190]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ypos = np.arange(len(labels))[::-1]
    cis = [1.96 * s for s in ses]
    ax.errorbar(betas, ypos, xerr=cis, fmt="o", color=BLACK, ecolor=BLACK,
                elinewidth=1.3, capsize=3, ms=6, mec="white", mew=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel("Coefficient (within-person, 95% CI)\npredicting next decision's contrarian status")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_horserace.pdf")
    plt.close(fig)


def fig3_category():
    # Coefficients with 95% CIs (person-clustered SEs), matching
    # table_category.tex, as forest plots.
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

    ax = axes[0]
    labels = ["Same category", "Different category"]
    betas = [0.069, 0.061]
    ses = [0.017, 0.014]
    ypos = np.arange(len(labels))[::-1]
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.errorbar(betas, ypos, xerr=[1.96 * s for s in ses], fmt="o", color=BLACK,
                ecolor=BLACK, elinewidth=1.3, capsize=3, ms=6, mec="white", mew=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel("Coefficient, 95% CI\n(recent contrarian rate -> next bet)")
    ax.set_title("(a) Next bet's category", fontsize=9)

    ax = axes[1]
    cats = ["Politics", "Sports", "Tech/AI", "Crypto/finance", "Science"]
    betas2 = [0.074, 0.092, 0.039, 0.063, 0.008]
    ses2 = [0.018, 0.027, 0.018, 0.026, 0.037]
    ypos2 = np.arange(len(cats))[::-1]
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.errorbar(betas2, ypos2, xerr=[1.96 * s for s in ses2], fmt="o", color=BLACK,
                ecolor=BLACK, elinewidth=1.3, capsize=3, ms=6, mec="white", mew=0.6)
    ax.set_yticks(ypos2)
    ax.set_yticklabels(cats, fontsize=8.5)
    ax.set_ylim(-0.7, len(cats) - 0.3)
    ax.set_xlabel("Coefficient, 95% CI, within category")
    ax.set_title("(b) Effect size by category", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_category.pdf")
    plt.close(fig)


def fig4_habit_model():
    # Calibrated against the corrected exact-K empirical decay curve (not the
    # earlier average-over-1..K measure). rho=0.998 has a mixing time of
    # ~1/(1-rho)=500 rounds, so burn-in and total simulation length are set
    # to several multiples of that (grid search over rho, sigma against the
    # empirical curve; see sim/calibrate_habit_model.py).
    cfg = Config(rho=0.998, sigma=0.013, n_rounds=7499)
    df = simulate(cfg, seed=1)
    df = add_recent_rate(df)
    q = df[df["round"] >= 3999].dropna(subset=["recent_rate"])
    K_list = [1, 5, 10, 20, 50, 100]
    r_model = []
    for K in K_list:
        qk = add_exact_k(q, K)
        _, r, _ = within_agent(qk, "recent_rate", f"exact_{K}")
        r_model.append(r)

    r_data = [0.0319, 0.0301, 0.0280, 0.0265, 0.0224, 0.0199]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.plot(K_list, r_data, marker="o", color=BLACK, lw=1.8, ms=5,
            mfc=BLACK, mec=BLACK, label="Empirical data")
    ax.plot(K_list, r_model, marker="s", color=MIDGRAY, lw=1.8, ms=5, ls="--",
            mfc="white", mec=MIDGRAY, mew=1.3, label="Habit model\n(outcome-blind)")
    ax.set_xscale("log")
    ax.set_xticks(K_list)
    ax.set_xticklabels([str(k) for k in K_list])
    ax.set_xlabel("Decisions ahead ($K$)")
    ax.set_ylabel("Correlation ($r$)")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_habit_model.pdf")
    plt.close(fig)


def fig5_feedback_eventtime():
    import pandas as pd
    csv_path = ROOT / "results" / "manifold_prototype" / "feedback_event_time.csv"
    s = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.axvline(-0.5, color="#cccccc", lw=1.0, ls="--", zorder=0)
    series = [
        (1.0, BLACK, "o", "-", BLACK, "Last bet resolved correct"),
        (0.0, MIDGRAY, "s", "--", "white", "Last bet resolved incorrect"),
    ]
    for label, color, marker, ls, mfc, name in series:
        sub = s[s.feedback_correct == label].sort_values("offset")
        ax.plot(sub.offset, sub["mean"], marker=marker, ms=4, lw=1.5, ls=ls,
                color=color, mfc=mfc, mec=color, mew=1.2, label=name)
        ax.fill_between(sub.offset, sub["mean"] - 1.96 * sub["sem"], sub["mean"] + 1.96 * sub["sem"],
                         color=color, alpha=0.13, lw=0)
    ax.set_xlabel("Decisions relative to feedback event\n(0 = first decision after resolution)")
    ax.set_ylabel("Contrarian rate")
    ax.legend(frameon=False, loc="center right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_feedback_eventtime.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig1_persistence()
    print("fig1 done")
    fig2_horserace()
    print("fig2 done")
    fig3_category()
    print("fig3 done")
    fig4_habit_model()
    print("fig4 done")
    fig5_feedback_eventtime()
    print("fig5 done")
