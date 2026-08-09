"""Replay the real trace through four autoscalers and price the result.

Scenarios
    static      one replica count, sized for the week's peak, running 24/7
    reactive    the Kubernetes HPA algorithm, target 60% CPU
    predictive  Prophet, 30-minute horizon, upper 80% bound
    hybrid      predictive with a reactive floor (scripts/scaling.recommend)

What is simulated
    A 10-second tick. Demand comes from the trace: the fleet-mean CPU u(t) of
    22 machines is read as a total load L(t) = u(t) x 22 "replica-percent", so
    running R replicas puts each of them at L(t)/R percent. Pods take
    --startup-s seconds to become Ready and carry no load until they do.

    The HPA loop follows the documented algorithm: it wakes every 15 s, reads a
    metric that is --metric-lag-s old, computes
    desired = ceil(currentReplicas x currentMetric / targetMetric), ignores the
    change if the ratio is within the 10% tolerance, applies the scale-down
    stabilisation window (300 s, take the *highest* recommendation in the
    window) and caps scale-up at max(4 pods, 100%) per 15 s.

Why this is not just "average replicas x price"
    The interesting difference between these controllers is not only what they
    cost, it is what they cost *at a given level of risk*. A controller that is
    cheap because it is late is not cheaper, it is worse. So the table reports
    saturated minutes and dropped load next to the dollars.

    python scripts/simulate.py --days 7
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

from dataio import figures_dir, load, quiet, results_dir
from scaling import recommend, required_replicas

warnings.filterwarnings("ignore")

TICK_S = 10
SYNC_S = 15
STAB_DOWN_S = 300
TOLERANCE = 0.10
N_REF = 22          # machines the fleet-mean was averaged over
SATURATION = 100.0  # per-replica CPU % at which no more work can be served
HOT = 90.0          # per-replica CPU % treated as an SLO breach


class Fleet:
    """Replica bookkeeping: pods are paid for immediately, useful later."""

    def __init__(self, replicas: int, startup_s: int):
        self.ready = replicas
        self.pending: list[float] = []   # ready-at times
        self.startup_s = startup_s
        self.actions = 0
        self.pods_started = 0

    @property
    def total(self) -> int:
        return self.ready + len(self.pending)

    def target(self, desired: int, now: float) -> None:
        if desired == self.total:
            return
        self.actions += 1
        if desired > self.total:
            n = desired - self.total
            self.pods_started += n
            self.pending.extend([now + self.startup_s] * n)
        else:
            # Terminate pending pods first -- they are not serving anything.
            n = self.total - desired
            drop = min(n, len(self.pending))
            self.pending = self.pending[:len(self.pending) - drop]
            self.ready -= (n - drop)

    def tick(self, now: float) -> None:
        due = [t for t in self.pending if t <= now]
        if due:
            self.ready += len(due)
            self.pending = [t for t in self.pending if t > now]


def build_forecasts(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                    refit_min: int, horizon_min: int, weekly: bool) -> pd.DataFrame:
    """Hourly retrain, walking forward. Every fit sees only its own past."""
    rows = []
    origins = pd.date_range(start, end, freq=f"{refit_min}min")
    for i, origin in enumerate(origins):
        history = df[df["ds"] <= origin].dropna(subset=["y"])
        m = Prophet(daily_seasonality=True, weekly_seasonality=weekly,
                    yearly_seasonality=False, interval_width=0.80)
        m.fit(history)
        stamps = pd.date_range(origin + pd.Timedelta(minutes=5),
                               origin + pd.Timedelta(minutes=refit_min + horizon_min),
                               freq="5min")
        fc = m.predict(pd.DataFrame({"ds": stamps}))
        fc["origin"] = origin
        rows.append(fc[["origin", "ds", "yhat", "yhat_upper"]])
        if (i + 1) % 24 == 0:
            print(f"  refit {i+1}/{len(origins)}")
    return pd.concat(rows, ignore_index=True)


class ForecastTape:
    """The forecasts a controller may look at, indexed for fast lookup.

    Only the freshest model whose origin is at or before `now` is visible --
    the simulation must never let a controller read a model trained on data it
    could not have had yet.
    """

    def __init__(self, fc: pd.DataFrame):
        self.origins = np.sort(fc["origin"].unique())
        self.by_origin = {
            o: (g["ds"].to_numpy(), g["yhat_upper"].to_numpy())
            for o, g in fc.sort_values("ds").groupby("origin")
        }

    def peak(self, now: pd.Timestamp, horizon_min: int) -> float | None:
        idx = np.searchsorted(self.origins, np.datetime64(now), side="right") - 1
        if idx < 0:
            return None
        stamps, upper = self.by_origin[self.origins[idx]]
        lo = np.searchsorted(stamps, np.datetime64(now), side="right")
        hi = np.searchsorted(
            stamps, np.datetime64(now + pd.Timedelta(minutes=horizon_min)), side="right")
        return float(upper[lo:hi].max()) if hi > lo else None


def simulate(scenario: str, demand: np.ndarray, times: pd.DatetimeIndex,
             fc: "ForecastTape | None", args) -> dict:
    """One controller over the whole window. Returns metrics + the replica trace."""
    target = args.target_cpu
    # hybrid_fast is the hybrid with its reactive floor evaluated at the HPA's
    # own cadence instead of the forecaster's. The forecast is still refreshed
    # hourly -- only the "what is it doing right now" half speeds up, and it is
    # a table lookup, so the extra decisions cost nothing.
    mode = "hybrid" if scenario == "hybrid_fast" else scenario
    decision_s = SYNC_S if scenario == "hybrid_fast" else args.decision_min * 60
    if scenario == "static":
        # "Enough pods for peak load, 24/7": size so the busiest 5-minute
        # bucket of the week still sits at the target utilisation.
        n = min(args.max_replicas,
                max(args.min_replicas, math.ceil(demand.max() / target)))
        start_replicas = n
    else:
        start_replicas = max(args.min_replicas, math.ceil(demand[0] / target))

    fleet = Fleet(start_replicas, args.startup_s)
    lag_ticks = args.metric_lag_s // TICK_S
    metric_history: list[float] = []                 # avg util over ready pods
    desired_history: deque[tuple[float, int]] = deque()   # scale-down window
    replicas_trace, util_trace = [], []
    saturated_ticks = hot_ticks = 0
    dropped = served = 0.0
    next_sync = 0.0
    next_decision = 0.0
    last_decision_driver = "forecast"
    driver_counts = {"forecast": 0, "observed": 0, "clamp": 0}

    for i, now in enumerate(np.arange(0, len(demand)) * TICK_S):
        fleet.tick(now)
        load = demand[i]
        util = load / max(1, fleet.ready)          # per-replica CPU %, uncapped
        # What the HPA would actually read. Two details from the algorithm:
        #   * a pod cannot report more than 100% of itself, so an overloaded
        #     fleet looks merely busy -- the overload is invisible upstream;
        #   * unready pods are excluded from the sum but, when scaling up, are
        #     assumed to consume 0%, i.e. the average is over *all* pods. That
        #     is the damping that stops a starting pod from being counted twice
        #     and sending the controller into a doubling spiral.
        metric_history.append(min(load, fleet.ready * SATURATION) / max(1, fleet.total))

        capacity = fleet.ready * SATURATION
        if load > capacity:
            saturated_ticks += 1
            dropped += (load - capacity) * TICK_S
            served += capacity * TICK_S
        else:
            served += load * TICK_S
        if util > HOT:
            hot_ticks += 1

        replicas_trace.append(fleet.total)
        util_trace.append(min(util, 150.0))

        if mode == "static":
            continue

        if mode == "reactive" and now >= next_sync:
            next_sync = now + SYNC_S
            # The controller reads a metric that is metric_lag_s old.
            if i >= lag_ticks:
                measured = metric_history[i - lag_ticks]
                ratio = measured / target
                if abs(ratio - 1.0) > TOLERANCE:
                    want = math.ceil(fleet.total * ratio)
                else:
                    want = fleet.total
                want = max(args.min_replicas, min(args.max_replicas, want))
                desired_history.append((now, want))
                fleet.target(_rate_limited(want, fleet.total, desired_history, now),
                             now)

        elif mode in ("predictive", "hybrid") and now >= next_decision:
            next_decision = now + decision_s
            peak = fc.peak(times[i], args.horizon_min)
            if peak is None:
                continue
            # The forecast is a fleet-mean over N_REF machines; convert it to
            # this deployment's replica count. ceil(R * (peak*N_REF/R)/target)
            # is just ceil(peak*N_REF/target), which is what we compute.
            predicted_at_current = peak * N_REF / max(1, fleet.total)
            observed = (metric_history[i - lag_ticks]
                        if mode == "hybrid" and i >= lag_ticks else None)
            d = recommend(fleet.total, predicted_at_current, observed_cpu=observed,
                          target_cpu_per_replica=target,
                          min_replicas=args.min_replicas,
                          max_replicas=args.max_replicas)
            driver_counts[d.driver] = driver_counts.get(d.driver, 0) + 1
            last_decision_driver = d.driver
            desired_history.append((now, d.recommended_replicas))
            fleet.target(_rate_limited(d.recommended_replicas, fleet.total,
                                       desired_history, now), now)

    minutes = len(demand) * TICK_S / 60
    replica_hours = float(np.sum(replicas_trace) * TICK_S / 3600)
    return {
        "scenario": scenario,
        "mean_replicas": float(np.mean(replicas_trace)),
        "min_replicas": int(np.min(replicas_trace)),
        "max_replicas": int(np.max(replicas_trace)),
        "replica_hours": replica_hours,
        "cost_usd": replica_hours * args.price_per_hour,
        "mean_utilisation": float(np.mean(util_trace)),
        "saturated_minutes": saturated_ticks * TICK_S / 60,
        "hot_minutes": hot_ticks * TICK_S / 60,
        "hot_pct_of_time": hot_ticks * TICK_S / 60 / minutes * 100,
        "dropped_load_pct": float(dropped / (dropped + served) * 100),
        "scaling_actions": fleet.actions,
        "pods_started": fleet.pods_started,
        "driver_counts": driver_counts,
        "_trace": replicas_trace,
        "_util": util_trace,
    }


def _rate_limited(want: int, current: int, history: "deque[tuple[float, int]]",
                  now: float) -> int:
    """Scale-up policy max(4 pods, 100%) per 15 s; scale-down only if the whole
    stabilisation window agrees (Kubernetes takes the highest recommendation in
    the window, so one quiet reading cannot shrink the deployment)."""
    while history and history[0][0] < now - STAB_DOWN_S:
        history.popleft()
    if want > current:
        return min(want, current + max(4, current))
    if want < current:
        return min(current, max(v for _, v in history) if history else want)
    return current


def build_demand(args):
    """The offered-load trace every scenario is replayed against."""
    raw = load(args.dataset)
    series = raw[["ds", "cpu"]].rename(columns={"cpu": "y"})
    # The controllers need a value every tick; the 5.2 h outage is filled by
    # interpolation *for the simulation only*. The forecaster still trains on
    # the gapped series, so it is not being handed data that never existed.
    filled = series.set_index("ds")["y"].interpolate(limit_direction="both")

    end = filled.index.max()
    start = end - pd.Timedelta(days=args.days)
    times = pd.date_range(start, end, freq=f"{TICK_S}s")
    u = filled.reindex(filled.index.union(times)).interpolate().reindex(times)
    demand = u.to_numpy() * N_REF     # total "replica-percent" of work

    flash = getattr(args, "flash_crowd", 0.0)
    if flash:
        # An unforecastable flash crowd: nothing in the history predicts it, so
        # every model in this repository will miss it. That is the point.
        at = pd.Timestamp(args.flash_at)
        mask = (times >= at) & (times < at + pd.Timedelta(minutes=args.flash_min))
        demand = demand.copy()
        demand[mask] *= flash
        print(f"flash crowd: x{flash} for {args.flash_min} min from {at} "
              f"({int(mask.sum())} ticks), peak now {demand.max()/N_REF:.1f}% fleet CPU")
    return series, times, demand, start, end


def main() -> None:
    quiet()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="alibaba")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--target-cpu", type=float, default=60.0)
    ap.add_argument("--min-replicas", type=int, default=2)
    ap.add_argument("--max-replicas", type=int, default=60)
    ap.add_argument("--startup-s", type=int, default=60)
    ap.add_argument("--metric-lag-s", type=int, default=60)
    ap.add_argument("--horizon-min", type=int, default=30)
    ap.add_argument("--decision-min", type=int, default=5)
    ap.add_argument("--refit-min", type=int, default=60)
    ap.add_argument("--price-per-hour", type=float, default=0.04)
    ap.add_argument("--no-weekly", action="store_true")
    ap.add_argument("--flash-crowd", type=float, default=0.0,
                    help="multiply demand by this factor for a short window")
    ap.add_argument("--flash-at", default="2018-01-05 14:00:00")
    ap.add_argument("--flash-min", type=int, default=20)
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    args = ap.parse_args()

    series, times, demand, start, end = build_demand(args)
    print(f"window {start} .. {end}  ({args.days} days, {len(times):,} ticks of {TICK_S}s)")
    print(f"demand: mean {demand.mean()/N_REF:.1f}% x {N_REF} machines, "
          f"peak {demand.max()/N_REF:.1f}%")
    print(f"\nfitting the forecaster every {args.refit_min} min "
          f"(each fit sees only its own past)...")
    fc_df = build_forecasts(series, start, end, args.refit_min, args.horizon_min,
                            weekly=not args.no_weekly)
    print(f"  {fc_df['origin'].nunique()} models, {len(fc_df):,} forecast points")
    fc = ForecastTape(fc_df)

    results = {}
    for scenario in ("static", "reactive", "predictive", "hybrid", "hybrid_fast"):
        r = simulate(scenario, demand, times, fc, args)
        results[scenario] = r
        print(f"\n== {scenario} ==")
        print(f"  replicas       mean {r['mean_replicas']:.1f}  "
              f"min {r['min_replicas']}  max {r['max_replicas']}")
        print(f"  cost           ${r['cost_usd']:.2f}  "
              f"({r['replica_hours']:.0f} replica-hours @ ${args.price_per_hour}/h)")
        print(f"  utilisation    mean {r['mean_utilisation']:.1f}% per replica")
        print(f"  risk           {r['hot_minutes']:.0f} min above {HOT:.0f}%, "
              f"{r['saturated_minutes']:.0f} min saturated, "
              f"{r['dropped_load_pct']:.3f}% of load dropped")
        print(f"  churn          {r['scaling_actions']} scaling actions, "
              f"{r['pods_started']} pods started")
        if r["driver_counts"].get("observed"):
            print(f"  reactive floor fired on {r['driver_counts']['observed']} "
                  f"of {sum(r['driver_counts'].values())} decisions")

    base = results["static"]["cost_usd"]
    print(f"\n{'scenario':12s} {'$/week':>9s} {'vs static':>10s} {'mean pods':>10s} "
          f"{'hot min':>8s} {'dropped %':>10s}")
    for k, r in results.items():
        print(f"{k:12s} {r['cost_usd']:9.2f} {(r['cost_usd']/base-1)*100:+9.1f}% "
              f"{r['mean_replicas']:10.1f} {r['hot_minutes']:8.0f} "
              f"{r['dropped_load_pct']:10.3f}")

    plot(results, times, demand, args)

    out = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
           for k, v in results.items()}
    out["_flash_crowd"] = {"factor": args.flash_crowd, "at": args.flash_at,
                           "minutes": args.flash_min}
    out["_assumptions"] = {
        "days": args.days, "target_cpu": args.target_cpu,
        "price_per_replica_hour": args.price_per_hour,
        "startup_s": args.startup_s, "metric_lag_s": args.metric_lag_s,
        "horizon_min": args.horizon_min, "decision_min": args.decision_min,
        "refit_min": args.refit_min, "tick_s": TICK_S, "sync_s": SYNC_S,
        "scale_down_stabilisation_s": STAB_DOWN_S, "tolerance": TOLERANCE,
        "hot_threshold_pct": HOT, "reference_machines": N_REF,
        "window": [str(start), str(end)],
    }
    path = results_dir() / f"simulation{args.tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


def plot(results: dict, times, demand, args) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    colours = {"static": "grey", "reactive": "tab:blue", "predictive": "tab:orange",
               "hybrid": "tab:green", "hybrid_fast": "tab:red"}
    for k, r in results.items():
        ax1.plot(times, r["_trace"], label=f"{k} (mean {r['mean_replicas']:.1f}, "
                 f"${r['cost_usd']:.0f}/week)", color=colours[k],
                 linewidth=1.0, alpha=0.9)
    ax1.set_ylabel("replicas")
    ax1.set_title(f"Replica count over {args.days} days of the Alibaba trace "
                  f"(target {args.target_cpu:.0f}% CPU, "
                  f"{args.startup_s}s pod start, {args.metric_lag_s}s metric lag)")
    ax1.legend(fontsize=8, loc="upper left", ncol=2)

    ax2.plot(times, demand / N_REF, color="black", linewidth=0.7, label="fleet CPU %")
    ax2.axhline(args.target_cpu, color="red", linestyle="--", alpha=0.5,
                label=f"target {args.target_cpu:.0f}%")
    ax2.set_ylabel("offered load\n(fleet CPU %)")
    ax2.set_xlabel("time")
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path = figures_dir() / f"simulation_replicas{args.tag}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
