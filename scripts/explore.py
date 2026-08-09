"""Step 2 of the lab: look at the data before modelling it.

Writes figures/metrics_overview.png and prints the three things the handout
asks you to look for -- repeating daily peaks, sudden spikes, and gaps.

    python scripts/explore.py --dataset alibaba
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataio import figures_dir, load, results_dir

SCALE_UP_THRESHOLD = 70.0


def describe(df: pd.DataFrame, name: str) -> dict:
    cpu = df["cpu"]
    span_days = (df["ds"].max() - df["ds"].min()).total_seconds() / 86400
    print(f"== {name} ==")
    print(df[["cpu", "memory"]].describe().round(2).to_string())
    print(f"\nrows          : {len(df)}   span: {span_days:.2f} days   step: 5 min")
    print(f"missing cpu   : {int(cpu.isna().sum())} "
          f"({cpu.isna().mean()*100:.1f}% of buckets)")

    # Gaps: contiguous runs of buckets nobody reported into.
    gaps = []
    if cpu.isna().any():
        miss = df.index[cpu.isna()].to_series()
        for _, run in miss.groupby((miss.diff() != 1).cumsum()):
            gaps.append((df.loc[run.iloc[0], "ds"], df.loc[run.iloc[-1], "ds"], len(run)))
        print("gaps          :")
        for start, end, n in gaps:
            print(f"  {start} -> {end}  ({n} buckets, {n*5/60:.1f} h)")

    # Seasonality evidence: autocorrelation at the daily lag.
    x = cpu.interpolate().to_numpy()
    acf = {}
    for lag, label in ((12, "1 h"), (144, "12 h"), (288, "24 h"), (2016, "7 d")):
        if lag < len(x) - 10:
            acf[label] = float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
    print("autocorrelation:", ", ".join(f"{k}={v:+.3f}" for k, v in acf.items()))

    hourly = cpu.groupby(df["ds"].dt.hour).mean()
    print(f"busiest hour-of-day: {int(hourly.idxmax())}:00 at {hourly.max():.1f}% CPU; "
          f"quietest: {int(hourly.idxmin())}:00 at {hourly.min():.1f}%")
    print(f"peak 5-min bucket  : {cpu.max():.1f}% at {df.loc[cpu.idxmax(), 'ds']}")
    print(f"minutes above {SCALE_UP_THRESHOLD:.0f}%: "
          f"{int((cpu > SCALE_UP_THRESHOLD).sum()) * 5} of {len(df)*5}\n")

    return {"rows": len(df), "span_days": round(span_days, 2),
            "missing_buckets": int(cpu.isna().sum()),
            "gaps": [[str(a), str(b), n] for a, b, n in gaps],
            "cpu_mean": round(float(cpu.mean()), 2),
            "cpu_std": round(float(cpu.std()), 2),
            "cpu_min": round(float(cpu.min()), 2),
            "cpu_max": round(float(cpu.max()), 2),
            "acf": {k: round(v, 3) for k, v in acf.items()},
            "busiest_hour": int(hourly.idxmax()),
            "quietest_hour": int(hourly.idxmin())}


def plot(df: pd.DataFrame, name: str, path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(df["ds"], df["cpu"], linewidth=0.8, color="steelblue")
    ax1.axhline(SCALE_UP_THRESHOLD, color="orange", linestyle="--", alpha=0.7,
                label=f"scale-up threshold ({SCALE_UP_THRESHOLD:.0f}%)")
    ax1.set_ylabel("CPU %")
    ax1.set_title(f"{name}: fleet-mean CPU utilisation (5-min buckets)"
                  if name == "alibaba" else f"{name}: CPU utilisation (5-min buckets)")
    ax1.legend(loc="upper left", fontsize=8)

    ax2.plot(df["ds"], df["memory"], linewidth=0.8, color="coral")
    ax2.set_ylabel("Memory %")
    ax2.set_title("Memory utilisation")
    ax2.set_xlabel("time")

    # Shade the gaps so a missing stretch cannot be mistaken for a flat one.
    if df["cpu"].isna().any():
        miss = df.index[df["cpu"].isna()].to_series()
        first = True
        for _, run in miss.groupby((miss.diff() != 1).cumsum()):
            a, b = df.loc[run.iloc[0], "ds"], df.loc[run.iloc[-1], "ds"]
            for ax in (ax1, ax2):
                ax.axvspan(a, b, color="grey", alpha=0.25,
                           label="no data" if first and ax is ax1 else None)
            first = False
        ax1.legend(loc="upper left", fontsize=8)

    # Mark the held-out 24 h so every later plot is easy to line up with this one.
    cutoff = df["ds"].max() - pd.Timedelta(hours=24)
    for ax in (ax1, ax2):
        ax.axvline(cutoff, color="black", linestyle=":", alpha=0.6)
    ax1.text(cutoff, ax1.get_ylim()[1] * 0.97, "  held-out 24 h ->",
             fontsize=8, va="top")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="alibaba", choices=["alibaba", "synthetic"])
    args = ap.parse_args()

    df = load(args.dataset)
    stats = describe(df, args.dataset)
    suffix = "" if args.dataset == "alibaba" else f"_{args.dataset}"
    plot(df, args.dataset, figures_dir() / f"metrics_overview{suffix}.png")

    import json
    (results_dir() / f"explore_{args.dataset}.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
