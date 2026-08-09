"""Steps 3-5 of the lab: fit Prophet, evaluate it, turn it into a replica count.

    python scripts/forecast_model.py --dataset alibaba
    python scripts/forecast_model.py --dataset synthetic --weekly off

Writes figures/cpu_forecast.png, figures/cpu_forecast_components.png,
figures/cpu_eval.png and results/forecast_<dataset>.json.

The evaluation here is the one the handout asks for: fit once, forecast the
whole held-out 24 hours. That is a *24-hour-ahead* forecast, which is not the
question an autoscaler asks -- see rolling_eval.py for the 30-minute-ahead
version, which is much more accurate and is what the scaling numbers use.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)

from dataio import as_prophet, figures_dir, load, quiet, results_dir, split
from scaling import recommend

SCALE_UP_THRESHOLD = 70.0


def fit(train: pd.DataFrame, weekly: bool) -> Prophet:
    m = Prophet(daily_seasonality=True,
                weekly_seasonality=weekly,
                yearly_seasonality=False,
                interval_width=0.80)
    m.fit(train)
    return m


def score(actual: np.ndarray, pred: np.ndarray,
          lower: np.ndarray | None = None,
          upper: np.ndarray | None = None) -> dict:
    out = {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, pred)),
        "mape": float(mean_absolute_percentage_error(actual, pred) * 100),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        # Signed error matters here in a way it does not for a generic forecast:
        # a model that is 3% low on average silently under-provisions.
        "bias": float(np.mean(pred - actual)),
        "under_forecast_pct": float(np.mean(pred < actual) * 100),
    }
    if lower is not None and upper is not None:
        inside = (actual >= lower) & (actual <= upper)
        out["interval_coverage_pct"] = float(inside.mean() * 100)
        out["mean_interval_width"] = float(np.mean(upper - lower))
        out["above_upper_pct"] = float(np.mean(actual > upper) * 100)
    return out


def plot_forecast(model: Prophet, forecast: pd.DataFrame, train: pd.DataFrame,
                  test: pd.DataFrame, dataset: str, path) -> None:
    fig = model.plot(forecast, figsize=(14, 5))
    ax = fig.gca()
    ax.plot(test["ds"], test["y"], color="crimson", linewidth=0.9,
            label="actual (held out, never seen in training)")
    ax.axvline(train["ds"].max(), color="black", linestyle=":", alpha=0.7)
    ax.set_title(f"Prophet CPU forecast - {dataset} "
                 f"(fit on {len(train)} points, {len(test)} held out)")
    ax.set_xlabel("time")
    ax.set_ylabel("CPU %")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def plot_components(model: Prophet, forecast: pd.DataFrame, path) -> None:
    fig = model.plot_components(forecast)
    fig.set_size_inches(10, 7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def plot_eval(test: pd.DataFrame, tf: pd.DataFrame, metrics: dict,
              dataset: str, path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(test["ds"], test["y"], label="actual", color="steelblue", linewidth=1.1)
    ax.plot(tf["ds"], tf["yhat"], label="predicted (yhat)", color="darkorange",
            linestyle="--", linewidth=1.3)
    ax.fill_between(tf["ds"], tf["yhat_lower"], tf["yhat_upper"], alpha=0.20,
                    color="orange", label="80% interval")
    ax.axhline(SCALE_UP_THRESHOLD, color="red", linestyle=":", alpha=0.6,
               label=f"{SCALE_UP_THRESHOLD:.0f}% scale-up threshold")

    # Mark where reality left the interval -- those are the minutes a
    # yhat_upper-based autoscaler would have under-provisioned.
    outside = test["y"].to_numpy() > tf["yhat_upper"].to_numpy()
    if outside.any():
        ax.scatter(test["ds"][outside], test["y"][outside], s=14, color="red",
                   zorder=5, label=f"above the interval ({int(outside.sum())} pts)")

    ax.set_title(f"Forecast vs actual on the held-out 24 h - {dataset}  "
                 f"(MAE {metrics['mae']:.2f} pp, MAPE {metrics['mape']:.1f}%, "
                 f"80% interval covers {metrics['interval_coverage_pct']:.0f}%)")
    ax.set_xlabel("time")
    ax.set_ylabel("CPU %")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    quiet()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="alibaba", choices=["alibaba", "synthetic"])
    ap.add_argument("--weekly", default="auto", choices=["on", "off", "auto"],
                    help="fit a weekly seasonality term. 'auto' picks the "
                         "variant with the lower MAE on a validation day -- the "
                         "24 h *before* the test day -- so the test set is never "
                         "used to choose a hyper-parameter.")
    ap.add_argument("--horizon-min", type=int, default=30)
    ap.add_argument("--current-replicas", type=int, default=20)
    ap.add_argument("--target-cpu", type=float, default=60.0)
    args = ap.parse_args()

    df = as_prophet(load(args.dataset))
    train, test = split(df)
    print(f"dataset {args.dataset}: {len(df)} usable points; "
          f"train {len(train)} ({train['ds'].min()} .. {train['ds'].max()}), "
          f"test {len(test)} ({test['ds'].min()} .. {test['ds'].max()})")

    results = {"dataset": args.dataset, "n_train": len(train), "n_test": len(test),
               "variants": {}}

    weekly_choice = args.weekly
    if weekly_choice == "auto":
        # Choose on a validation day carved out of the *training* data. Picking
        # the seasonality on the test day would be leakage: the reported MAPE
        # would then be the best of two tries rather than an honest forecast.
        sub_train, valid = split(train)
        val_scores = {}
        for weekly in (False, True):
            mv = fit(sub_train, weekly)
            fv = mv.predict(valid[["ds"]])
            val_scores[weekly] = mean_absolute_error(valid["y"], fv["yhat"])
        weekly_choice = "on" if val_scores[True] < val_scores[False] else "off"
        results["validation_mae"] = {"weekly_off": float(val_scores[False]),
                                     "weekly_on": float(val_scores[True])}
        print(f"validation day ({valid['ds'].min()} .. {valid['ds'].max()}): "
              f"MAE weekly_off={val_scores[False]:.3f}, "
              f"weekly_on={val_scores[True]:.3f} -> weekly={weekly_choice}")

    # Both seasonality variants, because with 8 days of data "add weekly
    # seasonality" is a claim that has to be checked, not a default to accept.
    for weekly in (False, True):
        m = fit(train, weekly)
        periods = len(test) + max(1, args.horizon_min // 5)
        future = m.make_future_dataframe(periods=periods, freq="5min")
        fc = m.predict(future)
        tf = fc[fc["ds"].isin(test["ds"])].reset_index(drop=True)
        assert len(tf) == len(test), f"alignment lost: {len(tf)} vs {len(test)}"
        met = score(test["y"].to_numpy(), tf["yhat"].to_numpy(),
                    tf["yhat_lower"].to_numpy(), tf["yhat_upper"].to_numpy())
        results["variants"]["weekly_on" if weekly else "weekly_off"] = met
        print(f"\n-- weekly_seasonality={weekly} --")
        for k, v in met.items():
            print(f"   {k:24s} {v:.3f}" if isinstance(v, float) else f"   {k:24s} {v}")

        if weekly == (weekly_choice == "on"):
            chosen = (m, fc, tf, met)

    m, fc, tf, met = chosen
    results["chosen_variant"] = f"weekly_{weekly_choice}"
    results["weekly_selection"] = args.weekly
    print(f"\nusing weekly_seasonality={weekly_choice} for the figures and the "
          f"recommendation")

    suffix = "" if args.dataset == "alibaba" else f"_{args.dataset}"
    plot_forecast(m, fc, train, test, args.dataset,
                  figures_dir() / f"cpu_forecast{suffix}.png")
    plot_components(m, fc, figures_dir() / f"cpu_forecast_components{suffix}.png")
    plot_eval(test, tf, met, args.dataset, figures_dir() / f"cpu_eval{suffix}.png")

    # What the components plot actually contains, in numbers -- a picture is not
    # a measurement, and the write-up needs the amplitudes.
    comp = {}
    if "daily" in fc:
        d = fc[["ds", "daily"]].copy()
        d["h"] = d["ds"].dt.hour
        hourly = d.groupby("h")["daily"].mean()
        comp["daily_amplitude_pp"] = float(fc["daily"].max() - fc["daily"].min())
        comp["daily_peak_hour"] = int(hourly.idxmax())
        comp["daily_trough_hour"] = int(hourly.idxmin())
    if "weekly" in fc:
        comp["weekly_amplitude_pp"] = float(fc["weekly"].max() - fc["weekly"].min())
    comp["trend_change_pp"] = float(fc["trend"].iloc[-1] - fc["trend"].iloc[0])
    comp["trend_start_pp"] = float(fc["trend"].iloc[0])
    comp["trend_end_pp"] = float(fc["trend"].iloc[-1])
    results["components"] = comp
    print("\ncomponents:", json.dumps(comp, indent=2))

    # Step 5: the recommendation, from the last `horizon` forecast points.
    horizon = fc.tail(max(1, args.horizon_min // 5))
    max_pred = float(horizon["yhat_upper"].max())
    observed = float(test["y"].iloc[-1])
    d = recommend(args.current_replicas, max_pred, observed_cpu=observed,
                  target_cpu_per_replica=args.target_cpu)
    print(f"\n--- Autoscaling recommendation ---")
    print(f"Current replicas:                 {d.current_replicas}")
    print(f"Max predicted CPU ({args.horizon_min} min):     {max_pred:.1f}% (upper 80% CI)")
    print(f"Last observed CPU:                {observed:.1f}%")
    print(f"Target CPU per replica:           {args.target_cpu:.0f}%")
    print(f"Recommended replicas:             {d.recommended_replicas} (driver: {d.driver})")
    print(f"DECISION: {d.action}")
    results["recommendation"] = {
        "current_replicas": d.current_replicas, "max_predicted_cpu": max_pred,
        "observed_cpu": observed, "recommended_replicas": d.recommended_replicas,
        "driver": d.driver, "action": d.action}

    path = results_dir() / f"forecast_{args.dataset}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
