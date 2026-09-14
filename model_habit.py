"""Minimal replacement model: shows that a purely outcome-blind, habit-like
process is SUFFICIENT to reproduce the observed persistence pattern (slow,
non-decaying-relative-to-noise decline in how strongly recent behavior
predicts future behavior, out to ~100 decisions). Unlike the earlier
self-attribution model, nothing here depends on whether past actions were
correct -- deliberately, since the empirical accuracy measures (recent and
lifetime) do not predict contrarian behavior once genuine behavioral
persistence is accounted for.

Each agent has a latent state z_t (propensity to defy the crowd) following a
simple, highly persistent autoregressive process with no feedback from
outcomes at all:
    z_{t+1} = rho * z_t + eps_t,   eps_t ~ N(0, sigma^2)
    P(contrarian_t = 1) = logistic(z_t)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    n_agents: int = 2000
    n_rounds: int = 7499
    rho: float = 0.998      # persistence of the latent state (close to 1 = sticky)
    sigma: float = 0.013    # innovation noise
    z0_sd: float = 0.5      # spread of initial dispositions across agents


def simulate(cfg: Config, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    N, T = cfg.n_agents, cfg.n_rounds
    z = rng.normal(0, cfg.z0_sd, size=N)

    rows = []
    for t in range(T):
        p = 1 / (1 + np.exp(-z))
        contrarian = (rng.random(N) < p).astype(np.int8)
        rows.append(pd.DataFrame({"agent": np.arange(N), "round": t,
                                   "z": z.copy(), "is_contrarian": contrarian}))
        z = cfg.rho * z + rng.normal(0, cfg.sigma, size=N)  # no dependence on `contrarian` or any outcome

    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    df = simulate(Config())
    print(f"simulated {df.agent.nunique()} agents x {df['round'].max()+1} rounds ({len(df):,} rows)")
