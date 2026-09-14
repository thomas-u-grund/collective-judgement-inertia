"""Recalibrate the habit model against the CORRECTED exact-K empirical decay
curve (r=+0.0319 at K=1 down to r=+0.0199 at K=100), replacing the earlier
calibration that was fit to an artifactually rising curve produced by the
average-over-1..K forward measure.

For each candidate rho, burn-in and total simulation length are set to a
generous multiple of the mixing time 1/(1-rho) to avoid the previously
diagnosed under-burn-in artifact.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model_habit import Config, simulate  # noqa: E402
from analyze_habit import add_recent_rate, add_exact_k, within_agent  # noqa: E402

K_LIST = [1, 5, 10, 20, 50, 100]
R_EMPIRICAL = [0.0319, 0.0301, 0.0280, 0.0265, 0.0224, 0.0199]


def run_config(rho, sigma, seed=1):
    mixing = 1 / (1 - rho)
    n_rounds = int(max(2000, mixing * 15))
    burn_in = int(max(500, mixing * 8))
    cfg = Config(rho=rho, sigma=sigma, n_rounds=n_rounds)
    df = simulate(cfg, seed=seed)
    df = add_recent_rate(df)
    q = df[df["round"] >= burn_in].dropna(subset=["recent_rate"])
    r_model = []
    for K in K_LIST:
        qk = add_exact_k(q, K)
        _, r, _ = within_agent(qk, "recent_rate", f"exact_{K}")
        r_model.append(r)
    return r_model, n_rounds, burn_in


if __name__ == "__main__":
    candidates = [
        (0.997, 0.013), (0.997, 0.014), (0.998, 0.013), (0.998, 0.014),
        (0.9985, 0.012), (0.9985, 0.013),
    ]
    print(f"{'rho':>7} {'sigma':>7}  " + "  ".join(f"K={k:<4}" for k in K_LIST) + "   sse")
    print(f"{'empir':>7} {'':>7}  " + "  ".join(f"{r:+.4f}" for r in R_EMPIRICAL))
    best = None
    for rho, sigma in candidates:
        r_model, n_rounds, burn_in = run_config(rho, sigma)
        sse = sum((a - b) ** 2 for a, b in zip(r_model, R_EMPIRICAL))
        print(f"{rho:7.3f} {sigma:7.3f}  " + "  ".join(f"{r:+.4f}" for r in r_model) +
              f"   {sse:.6f}   (n_rounds={n_rounds}, burn_in={burn_in})")
        if best is None or sse < best[0]:
            best = (sse, rho, sigma, n_rounds, burn_in)
    print(f"\nBest fit: rho={best[1]}, sigma={best[2]}, n_rounds={best[3]}, burn_in={best[4]}, sse={best[0]:.6f}")
