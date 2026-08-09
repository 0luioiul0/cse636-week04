"""Shared loading + train/test splitting, so every script splits identically."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "alibaba": ROOT / "data" / "alibaba_5min.csv",
    "synthetic": ROOT / "data" / "synthetic_5min.csv",
}
TEST_HOURS = 24  # the handout's held-out window


def quiet() -> None:
    """Silence Stan's per-fit chatter.

    Prophet emits two INFO lines through cmdstanpy for every fit. The rolling
    evaluation fits ~50 times and the simulation ~170, which buries the output
    that matters. Call this *after* prophet has been imported -- importing it
    re-enables the logger.
    """
    for name in ("cmdstanpy", "prophet"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.disabled = True
        lg.propagate = False


def load(name: str = "alibaba") -> pd.DataFrame:
    df = pd.read_csv(DATASETS[name], parse_dates=["ds"])
    return df


def as_prophet(df: pd.DataFrame, column: str = "cpu") -> pd.DataFrame:
    """Prophet wants exactly ds/y. Rows with no reading are dropped, not filled:
    Prophet regresses on time, so a hole in the series is simply an absence of
    evidence -- inventing values there would train the model on our own guess."""
    out = df[["ds", column]].rename(columns={column: "y"})
    return out.dropna(subset=["y"]).reset_index(drop=True)


def split(df: pd.DataFrame, hours: int = TEST_HOURS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split -- never shuffled: the test set must lie in the future."""
    cutoff = df["ds"].max() - pd.Timedelta(hours=hours)
    return (df[df["ds"] < cutoff].copy().reset_index(drop=True),
            df[df["ds"] >= cutoff].copy().reset_index(drop=True))


def figures_dir() -> Path:
    d = ROOT / "figures"
    d.mkdir(exist_ok=True)
    return d


def results_dir() -> Path:
    d = ROOT / "results"
    d.mkdir(exist_ok=True)
    return d
