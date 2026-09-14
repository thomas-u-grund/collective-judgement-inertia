"""Shared configuration for the manifold_prototype analysis scripts.

Set the MANIFOLD_DATA_DIR environment variable to point at your local copy
of the Manifold Markets bulk data export (the directory containing
`bets_dataset`, `contracts_dataset`, and `contracts_groupSlugs.csv`). If
unset, defaults to a `data/manifold` directory alongside this repository,
which is not included here -- see the README for how to obtain the data.
"""
import os
from pathlib import Path

DK_MANIFOLD = Path(os.environ.get(
    "MANIFOLD_DATA_DIR",
    Path(__file__).resolve().parent.parent / "data" / "manifold",
))
