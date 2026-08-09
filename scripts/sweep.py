"""Under what conditions does predicting beat reacting?

The headline simulation answers "on this workload, with a 60-second pod start,
reactive HPA is cheaper and never breaches". That is one point in a parameter
space, and quoting it as the answer would be as misleading as quoting the
opposite. This sweeps the two parameters the lecture claims are decisive --
how long a pod takes to become useful, and how sharp the load step is -- and
reports where each controller stops working.

    python scripts/sweep.py                 # startup sweep + flash-crowd sweep

Forecasts are built once and reused: they do not depend on pod startup time.
"""
from __future__ import annotations

import argparse
import copy
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dataio import figures_dir, quiet, results_dir
from simulate import ForecastTape, build_demand, build_forecasts, simulate

SCENARIOS = ("static", "reactive", "predictive", "hybrid", "hybrid_fast")


class Args:
    """Plain settings object -- simulate() only reads attributes."""
    dataset = "alibaba"
    days = 7
    target_cpu = 60.0
    min_replicas = 2
    max_replicas = 60
    startup_s = 60
    metric_lag_s = 60
    horizon_min = 30
    decision_min = 5
    refit_min = 60
    price_per_hour = 0.04
    no_weekly = False
    flash_crowd = 0.0
    flash_at = "2018-01-05 14:00:00"
    flash_min = 20


def run(args, times, demand, tape) -> dict:
    return {s: {k: v for k, v in simulate(s, demand, times, tape, args).items()
                if not k.startswith("_")}
            for s in SCENARIOS}


def main() -> None:
    quiet()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--startups", type=int, nargs="*",
                    default=[15, 60, 150, 300, 600])
    ap.add_argument("--flash-factors", type=float, nargs="*",
                    default=[1.0, 1.5, 2.0, 2.5])
    args0 = ap.parse_args()

    base = Args()
    series, times, demand, start, end = build_demand(base)
    print("fitting the forecaster once (it does not depend on pod startup)...")
    tape = ForecastTape(build_forecasts(series, start, end, base.refit_min,
                                        base.horizon_min, weekly=True))

    out = {"startup_sweep": {}, "flash_sweep": {}, "_base": vars(Args) and {
        k: getattr(Args, k) for k in
        ("days", "target_cpu", "metric_lag_s", "horizon_min", "decision_min",
         "refit_min", "price_per_hour", "min_replicas", "max_replicas")}}

    print(f"\n=== pod startup sweep (flash crowd off) ===")
    print(f"{'startup':>8s}  {'scenario':11s} {'$/week':>8s} {'mean pods':>10s} "
          f"{'hot min':>8s} {'sat min':>8s} {'dropped %':>10s} {'pods started':>13s}")
    for s in args0.startups:
        a = copy.copy(base)
        a.startup_s = s
        res = run(a, times, demand, tape)
        out["startup_sweep"][s] = res
        for name in SCENARIOS:
            r = res[name]
            print(f"{s:6d}s  {name:11s} {r['cost_usd']:8.2f} {r['mean_replicas']:10.1f} "
                  f"{r['hot_minutes']:8.0f} {r['saturated_minutes']:8.0f} "
                  f"{r['dropped_load_pct']:10.3f} {r['pods_started']:13d}")
        print()

    print(f"=== flash-crowd sweep (pod startup {base.startup_s}s, "
          f"{base.flash_min} min at {base.flash_at}) ===")
    print(f"{'factor':>7s}  {'scenario':11s} {'$/week':>8s} {'hot min':>8s} "
          f"{'sat min':>8s} {'dropped %':>10s} {'recovery':>9s}")
    for f in args0.flash_factors:
        a = copy.copy(base)
        a.flash_crowd = 0.0 if f == 1.0 else f
        _, times_f, demand_f, _, _ = build_demand(a)
        res = run(a, times_f, demand_f, tape)
        out["flash_sweep"][f] = res
        for name in SCENARIOS:
            r = res[name]
            print(f"{f:7.1f}  {name:11s} {r['cost_usd']:8.2f} {r['hot_minutes']:8.0f} "
                  f"{r['saturated_minutes']:8.0f} {r['dropped_load_pct']:10.3f} "
                  f"{r['saturated_minutes']:8.1f}m")
        print()

    plot_startup(out["startup_sweep"], args0.startups)
    plot_flash(out["flash_sweep"], args0.flash_factors)
    path = results_dir() / "sweep.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def plot_startup(sweep: dict, startups: list[int]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    colours = {"static": "grey", "reactive": "tab:blue", "predictive": "tab:orange",
               "hybrid": "tab:green", "hybrid_fast": "tab:red"}
    for name in SCENARIOS:
        hot = [sweep[s][name]["hot_minutes"] for s in startups]
        cost = [sweep[s][name]["cost_usd"] for s in startups]
        ax1.plot(startups, hot, "o-", color=colours[name], label=name)
        ax2.plot(startups, cost, "o-", color=colours[name], label=name)
    ax1.set_xlabel("pod startup time (s)")
    ax1.set_ylabel("minutes above 90% CPU per replica")
    ax1.set_title("Risk: how long the service runs hot")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("pod startup time (s)")
    ax2.set_ylabel("$ per week")
    ax2.set_title("Cost")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir() / "sweep_startup.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def plot_flash(sweep: dict, factors: list[float]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    colours = {"static": "grey", "reactive": "tab:blue", "predictive": "tab:orange",
               "hybrid": "tab:green", "hybrid_fast": "tab:red"}
    for name in SCENARIOS:
        sat = [sweep[f][name]["saturated_minutes"] for f in factors]
        drop = [sweep[f][name]["dropped_load_pct"] for f in factors]
        ax1.plot(factors, sat, "o-", color=colours[name], label=name)
        ax2.plot(factors, drop, "o-", color=colours[name], label=name)
    ax1.set_xlabel("flash-crowd multiplier (20 min, unforecastable)")
    ax1.set_ylabel("minutes at 100% CPU (no capacity left)")
    ax1.set_title("A spike no model predicted")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("flash-crowd multiplier")
    ax2.set_ylabel("% of offered load dropped over the week")
    ax2.set_title("Work that could not be served")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir() / "sweep_flash.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
