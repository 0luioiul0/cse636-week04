"""Turn the raw Alibaba rows into the two 5-minute series this lab forecasts.

Outputs
    data/alibaba_5min.csv    real   -- fleet-mean CPU and memory, 8 days, 5-min grid
    data/synthetic_5min.csv  control -- the handout's generator verbatim (seed 42)

Why a *fleet mean* and not one machine
    A Kubernetes HPA does not look at one pod; it compares the *average*
    utilisation across the pods of a Deployment against a target. Averaging the
    serving machines reproduces the signal the autoscaler actually sees, and it
    is also what makes the replica arithmetic in scaling.py meaningful.

Why two machines are dropped
    m_1933 and m_1946 sit at ~5% CPU for the whole window while the other 22
    average 32-47%. They are a different workload class, not a quiet moment of
    the same one; averaging them in would drag the level down without changing
    the shape. The exclusion is printed, and --keep-idle turns it off.

On timestamps
    time_stamp is seconds from an origin the trace does not disclose. It is
    mapped to 2018-01-01 00:00 so pandas has a DatetimeIndex. Clock-time and
    weekday labels are therefore arbitrary: the *shape* of the daily cycle is
    real data, "the peak is at 15:00" is not a claim about anybody's local time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ORIGIN = pd.Timestamp("2018-01-01 00:00:00")
IDLE_CPU_THRESHOLD = 10.0  # mean CPU % below which a machine is a different workload


def prepare_alibaba(raw: Path, out: Path, keep_idle: bool = False) -> pd.DataFrame:
    df = pd.read_csv(raw)
    df["ds"] = ORIGIN + pd.to_timedelta(df["time_stamp"], unit="s")

    per_machine = df.groupby("machine_id")["cpu_util_percent"].mean()
    idle = sorted(per_machine[per_machine < IDLE_CPU_THRESHOLD].index)
    print(f"machines in file        : {per_machine.size}")
    print(f"near-idle (mean CPU <{IDLE_CPU_THRESHOLD:.0f}%): {idle or 'none'}")
    if idle and not keep_idle:
        df = df[~df["machine_id"].isin(idle)]
        print(f"kept                    : {df['machine_id'].nunique()} serving machines")

    # Raw sampling is nominally 10 s but is really ~60 s with long dropouts, so
    # resample each machine onto the 5-minute grid first, then average. Doing it
    # the other way round would let a machine that happens to report twice in a
    # bucket outvote one that reported once.
    grid = []
    for mid, g in df.groupby("machine_id"):
        s = (g.set_index("ds")[["cpu_util_percent", "mem_util_percent"]]
               .resample("5min").mean())
        s.columns = pd.MultiIndex.from_product([[mid], s.columns])
        grid.append(s)
    wide = pd.concat(grid, axis=1)

    cpu = wide.xs("cpu_util_percent", axis=1, level=1)
    mem = wide.xs("mem_util_percent", axis=1, level=1)
    out_df = pd.DataFrame({
        "ds": cpu.index,
        "cpu": cpu.mean(axis=1).to_numpy(),
        "memory": mem.mean(axis=1).to_numpy(),
        "machines_reporting": cpu.notna().sum(axis=1).to_numpy(),
    })

    total = len(out_df)
    empty = int((out_df["machines_reporting"] == 0).sum())
    thin = int(((out_df["machines_reporting"] > 0) &
                (out_df["machines_reporting"] < cpu.shape[1] / 2)).sum())
    print(f"5-min buckets           : {total} "
          f"({(out_df['ds'].max()-out_df['ds'].min()).total_seconds()/86400:.2f} days)")
    print(f"buckets with no reading : {empty}")
    print(f"buckets under half the fleet reporting: {thin}")

    out_df.to_csv(out, index=False)
    print(f"wrote {out}")
    return out_df


def prepare_synthetic(out: Path) -> pd.DataFrame:
    """The lab handout's generator, unmodified apart from the deprecated '5T'."""
    np.random.seed(42)
    n_points = 2016  # 7 days at 5-minute intervals
    timestamps = pd.date_range(start="2025-10-01", periods=n_points, freq="5min")

    t = np.arange(n_points)
    daily_cycle = 20 * np.sin(2 * np.pi * t / (24 * 12))
    weekly_cycle = 10 * np.sin(2 * np.pi * t / (7 * 24 * 12))
    noise = np.random.normal(0, 3, n_points)
    trend = 0.005 * t

    cpu = np.clip(30 + daily_cycle + weekly_cycle + noise + trend, 5, 95)
    memory = np.clip(45 + 0.5 * daily_cycle + noise * 0.5, 20, 90)

    df = pd.DataFrame({"ds": timestamps, "cpu": cpu, "memory": memory,
                       "machines_reporting": 1})
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows, seed 42)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/alibaba_raw.csv")
    ap.add_argument("--keep-idle", action="store_true")
    args = ap.parse_args()

    data = Path("data")
    data.mkdir(exist_ok=True)
    print("== real: Alibaba cluster-trace-v2018 ==")
    prepare_alibaba(Path(args.raw), data / "alibaba_5min.csv", args.keep_idle)
    print("\n== control: the handout's synthetic generator ==")
    prepare_synthetic(data / "synthetic_5min.csv")


if __name__ == "__main__":
    main()
