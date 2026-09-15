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

# Validated two-series categorical palette (dataviz skill reference palette,
# slots 1-2: blue/orange, CVD Delta E 9.1+ adjacent, normal-vision 19.6+).
MANIFOLD_COLOR = "#2a78d6"
POLYMARKET_COLOR = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.labelcolor": "#222222",
    "xtick.color": "#555555", "ytick.color": "#555555", "figure.dpi": 300,
})


def fig1_persistence():
    # Exact-K correlation -- the contrarian status of the decision exactly
    # K steps ahead, not an average over decisions 1..K -- computed
    # identically (recent-contrarian-rate over the past 20 decisions, built
    # from placement order irrespective of resolution status, within-person
    # demeaned Pearson r) on each dataset's own final qualifying panel (same
    # panel used for the joint model in Fig. 2), in Manifold and Polymarket.
    # The two platforms are now within a similar order of magnitude, so a
    # shared linear y-axis is used (no longer log).
    # (A third dataset, the Good Judgment Project, was also tested; its
    # persistence result does not survive a first-encounter-per-question
    # restriction and its consensus was not visible to most participants,
    # so it is reported only as a structural check, Supplementary Note 1.)
    manifold_K = [1, 5, 10, 20, 50, 100]
    manifold_r = [0.2065, 0.1608, 0.1351, 0.1109, 0.0763, 0.0546]

    poly_K = [1, 2, 5, 10, 25, 50, 100]
    poly_r = [0.27213, 0.24515, 0.19979, 0.16383, 0.12106, 0.09377, 0.07411]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(manifold_K, manifold_r, marker="o", color=MANIFOLD_COLOR, lw=1.8, ms=5,
            mfc=MANIFOLD_COLOR, mec="white", mew=0.6, label="Manifold Markets\n(play-money, 2021-2024)")
    ax.plot(poly_K, poly_r, marker="s", color=POLYMARKET_COLOR, lw=1.8, ms=5,
            mfc=POLYMARKET_COLOR, mec="white", mew=0.6, label="Polymarket\n(real-money, 2020-2026)")
    ax.set_xscale("log")
    ax.set_ylim(0, 0.30)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1", "10", "100"])
    ax.set_xlabel("Decisions ahead ($K$)")
    ax.set_ylabel("Correlation ($r$)")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)

    fig.tight_layout(pad=1.2)
    fig.savefig(FIGDIR / "fig1_persistence.pdf")
    plt.close(fig)


def fig2_horserace():
    # Coefficients (not t-statistics) with 95% CIs, from the identical
    # joint within-person model (recent accuracy, baseline accuracy,
    # recent contrarian rate jointly predicting current contrarian
    # status; person/wallet-clustered SEs), estimated separately on each
    # dataset's own final qualifying panel (same panel as Fig. 1).
    # Combined on one shared x-axis, colored by dataset, with a small
    # vertical offset per category so the two datasets' CIs don't
    # overlap; the shared axis makes the real magnitude gap between
    # platforms visible rather than hiding it behind separate scales.
    labels = ["Recent\naccuracy", "Baseline\naccuracy", "Recent\ncontrarian rate"]
    manifold = ([0.01232, 0.03990, 0.57448], [0.00489, 0.02210, 0.01508])
    polymarket = ([-0.00761, -0.03373, 0.63196], [0.00192, 0.01067, 0.00685])

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ypos = np.arange(len(labels))[::-1]
    offset = 0.14

    betas, ses = manifold
    ax.errorbar(betas, ypos + offset, xerr=[1.96 * s for s in ses], fmt="o",
                color=MANIFOLD_COLOR, ecolor=MANIFOLD_COLOR, elinewidth=1.3, capsize=3,
                ms=6, mec="white", mew=0.6, label="Manifold Markets")
    betas, ses = polymarket
    ax.errorbar(betas, ypos - offset, xerr=[1.96 * s for s in ses], fmt="s",
                color=POLYMARKET_COLOR, ecolor=POLYMARKET_COLOR, elinewidth=1.3, capsize=3,
                ms=6, mec="white", mew=0.6, label="Polymarket")

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel("Coefficient, 95% CI")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)

    fig.tight_layout(pad=1.2)
    fig.savefig(FIGDIR / "fig2_horserace.pdf")
    plt.close(fig)


def fig3_category():
    # Coefficients with 95% CIs (person/wallet-clustered SEs), matching
    # table_category.tex, as forest plots, in Manifold and Polymarket
    # (both have genuinely heterogeneous topic/market-type content and a
    # keyword-based classification; Methods). Combined per panel, colored
    # by dataset, small vertical offset per category. The Good Judgment
    # Project is not included: its questions are, by the tournament's own
    # design, uniformly geopolitical, so no non-arbitrary topic split
    # analogous to the other two platforms exists (Supplementary Note 12).
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    offset = 0.14

    def combined_panel(ax, labels, m_betas, m_ses, p_betas, p_ses, title, legend=False):
        ypos = np.arange(len(labels))[::-1]
        ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
        ax.errorbar(m_betas, ypos + offset, xerr=[1.96 * s for s in m_ses], fmt="o",
                    color=MANIFOLD_COLOR, ecolor=MANIFOLD_COLOR, elinewidth=1.3, capsize=3,
                    ms=6, mec="white", mew=0.6, label="Manifold Markets")
        ax.errorbar(p_betas, ypos - offset, xerr=[1.96 * s for s in p_ses], fmt="s",
                    color=POLYMARKET_COLOR, ecolor=POLYMARKET_COLOR, elinewidth=1.3, capsize=3,
                    ms=6, mec="white", mew=0.6, label="Polymarket")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_ylim(-0.7, len(labels) - 0.3)
        ax.set_xlabel("Coefficient, 95% CI", fontsize=8.5)
        ax.set_title(title, fontsize=9)
        if legend:
            ax.legend(frameon=False, loc="lower right", fontsize=7.5)

    combined_panel(
        axes[0], ["Same category", "Different category"],
        [0.566, 0.304], [0.0146, 0.0181],
        [0.597, 0.277], [0.0113, 0.0141],
        "(a) Next decision's category", legend=True,
    )
    combined_panel(
        axes[1], ["Politics", "Sports", "Tech/AI", "Crypto/finance", "Science"],
        [0.556, 0.547, 0.514, 0.563, 0.457], [0.0242, 0.0180, 0.0307, 0.0287, 0.0459],
        [0.620, 0.568, 0.532, 0.627, 0.614], [0.0122, 0.0149, 0.0196, 0.0100, 0.0487],
        "(b) Effect size by category",
    )

    fig.tight_layout(pad=1.2)
    fig.savefig(FIGDIR / "fig3_category.pdf")
    plt.close(fig)


def fig4_habit_model():
    # Calibrated against the exact-K empirical decay curve computed with
    # recent contrarian rate built from placement order (irrespective of
    # resolution status), matching GJP's and Polymarket's construction.
    # rho=0.986 has a mixing time of ~1/(1-rho)=71 rounds; burn-in and total
    # simulation length are set to several multiples of that (grid search
    # over rho, sigma against the empirical curve; see
    # sim/calibrate_habit_model.py).
    cfg = Config(rho=0.986, sigma=0.100, n_rounds=2000)
    df = simulate(cfg, seed=1)
    df = add_recent_rate(df)
    q = df[df["round"] >= 571].dropna(subset=["recent_rate"])
    K_list = [1, 5, 10, 20, 50, 100]
    r_model = []
    for K in K_list:
        qk = add_exact_k(q, K)
        _, r, _ = within_agent(qk, "recent_rate", f"exact_{K}")
        r_model.append(r)

    r_data = [0.2065, 0.1608, 0.1351, 0.1109, 0.0763, 0.0546]
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
