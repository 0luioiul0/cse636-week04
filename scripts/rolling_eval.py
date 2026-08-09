"""The evaluation an autoscaler actually needs: 30 minutes ahead, refit as you go.

The handout's evaluation fits once and forecasts the whole held-out day. No
autoscaler works that way -- it re-decides every few minutes with all the
history up to *now*, and it only ever looks one short horizon ahead. Those two
questions have very different answers, and quoting the 24-hour number as though
it were the autoscaling accuracy understates the method badly.

For every origin on the test day (default: every 30 minutes) this script
retrains on everything up to that origin and forecasts the next 30 minutes, so
each held-out point is scored exactly once at a horizon of 5-30 minutes. The
same origins are scored with four cheaper baselines, so "Prophet is better" is a
measurement rather than an assumption.

It also scores the *decision*, not just the curve: for each origin, does
max(yhat_upper) over the next 30 minutes cover the actual maximum? A forecast
that is accurate on average but misses every peak is useless for scale-up.

    python scripts/rolling_eval.py --dataset alibaba
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)

from dataio import as_prophet, load, quiet, results_dir, split

warnings.filterwarnings("ignore")
STEP_MIN = 5


def prophet_forecast(history: pd.DataFrame, stamps: pd.Series, weekly: bool,
                     level_window: int = 6):
    """Fit, forecast, and also return a *level-adjusted* variant.

    Prophet has no autoregressive term: yhat(t) is a function of the clock, not
    of what the metric is doing right now. So when the fleet sits 8 points above
    the seasonal curve for an hour, Prophet keeps predicting the curve. The
    adjustment is the smallest possible repair -- add the mean residual over the
    last `level_window` observations (30 minutes) to the whole forecast.
    """
    m = Prophet(daily_seasonality=True, weekly_seasonality=weekly,
                yearly_seasonality=False, interval_width=0.80)
    m.fit(history)
    recent = history.tail(level_window)
    fitted = m.predict(recent[["ds"]])["yhat"].to_numpy()
    offset = float(np.mean(recent["y"].to_numpy() - fitted))
    fc = m.predict(pd.DataFrame({"ds": stamps}))
    return (fc["yhat"].to_numpy(), fc["yhat_upper"].to_numpy(), offset)


def arima_forecast(history: pd.DataFrame, k: int, window: int = 288):
    """ARIMA on a trailing window. A *seasonal* ARIMA would need m=288 for the
    daily cycle at 5-minute resolution, which statsmodels will not fit in
    reasonable time -- that limitation is itself part of the comparison."""
    from statsmodels.tsa.arima.model import ARIMA
    y = history["y"].to_numpy()[-window:]
    res = ARIMA(y, order=(2, 1, 2)).fit()
    f = res.get_forecast(steps=k)
    return f.predicted_mean, f.conf_int(alpha=0.20)[:, 1]


def score(actual: np.ndarray, pred: np.ndarray) -> dict:
    return {"mae": float(mean_absolute_error(actual, pred)),
            "mape": float(mean_absolute_percentage_error(actual, pred) * 100),
            "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
            "bias": float(np.mean(pred - actual))}


def main() -> None:
    quiet()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="alibaba", choices=["alibaba", "synthetic"])
    ap.add_argument("--horizon-min", type=int, default=30)
    ap.add_argument("--refit-min", type=int, default=30)
    ap.add_argument("--weekly", action="store_true", default=True)
    ap.add_argument("--skip-arima", action="store_true")
    args = ap.parse_args()

    df = as_prophet(load(args.dataset))
    train, test = split(df)
    k = args.horizon_min // STEP_MIN
    origins = test["ds"][::args.refit_min // STEP_MIN].tolist()
    print(f"{args.dataset}: {len(origins)} origins, horizon {args.horizon_min} min "
          f"({k} steps of {STEP_MIN} min)")

    lookup = df.set_index("ds")["y"]
    rows, decisions = [], []
    t0 = time.time()

    for i, origin in enumerate(origins):
        stamps = pd.date_range(origin + pd.Timedelta(minutes=STEP_MIN),
                               periods=k, freq=f"{STEP_MIN}min")
        actual = lookup.reindex(stamps)
        if actual.isna().all():
            continue
        history = df[df["ds"] <= origin]
        valid = ~actual.isna().to_numpy()
        a = actual.to_numpy()[valid]

        yhat, yupper, offset = prophet_forecast(history, pd.Series(stamps), args.weekly)
        preds = {"prophet": yhat[valid],
                 "prophet_level_adj": (yhat + offset)[valid]}

        # --- baselines, all using only data up to the same origin -----------
        last = float(history["y"].iloc[-1])
        preds["persistence"] = np.full(valid.sum(), last)
        preds["moving_avg_30m"] = np.full(valid.sum(), float(history["y"].tail(k).mean()))
        seasonal = lookup.reindex(stamps - pd.Timedelta(hours=24)).to_numpy()[valid]
        preds["seasonal_naive_24h"] = seasonal

        if not args.skip_arima:
            am, au = arima_forecast(history, k)
            preds["arima_212"] = am[valid]

        for name, p in preds.items():
            ok = ~np.isnan(p)
            for j, (act, pr) in enumerate(zip(a[ok], p[ok])):
                rows.append({"origin": origin, "method": name,
                             "horizon_min": (j + 1) * STEP_MIN,
                             "actual": act, "pred": pr})

        # --- the decision-level question -----------------------------------
        decisions.append({"origin": str(origin),
                          "actual_max": float(a.max()),
                          "prophet_upper_max": float(yupper[valid].max()),
                          "prophet_point_max": float(yhat[valid].max()),
                          "level_adj_upper_max": float((yupper + offset)[valid].max()),
                          "level_offset": offset,
                          "observed_now": last,
                          "persistence_max": last})

        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(origins)} origins, {time.time()-t0:.0f}s")

    long = pd.DataFrame(rows)
    print(f"\nscored {len(long)} (origin, method, step) points in {time.time()-t0:.0f}s\n")

    summary = {}
    for name, g in long.groupby("method"):
        summary[name] = score(g["actual"].to_numpy(), g["pred"].to_numpy())
        summary[name]["n"] = int(len(g))
    order = sorted(summary, key=lambda k_: summary[k_]["mae"])
    print(f"{'method':20s} {'MAE (pp)':>9s} {'MAPE %':>8s} {'RMSE':>7s} {'bias':>7s}")
    for name in order:
        s = summary[name]
        print(f"{name:20s} {s['mae']:9.2f} {s['mape']:8.1f} {s['rmse']:7.2f} {s['bias']:+7.2f}")

    # Error growth with horizon -- the reason short horizons are worth having.
    by_h = (long[long.method == "prophet"]
            .groupby("horizon_min")
            .apply(lambda g: mean_absolute_error(g["actual"], g["pred"]),
                   include_groups=False))
    print("\nProphet MAE by horizon (min):",
          ", ".join(f"{int(h)}m={v:.2f}" for h, v in by_h.items()))

    dec = pd.DataFrame(decisions)
    coverage = {}
    print(f"\nDecision-level: does the upper bound cover the actual 30-min peak?")
    for label, col in (("prophet yhat_upper", "prophet_upper_max"),
                       ("level-adjusted upper", "level_adj_upper_max"),
                       ("reactive floor (max of upper, now)", None)):
        if col is None:
            bound = dec[["level_adj_upper_max", "observed_now"]].max(axis=1)
        else:
            bound = dec[col]
        covered = float((dec["actual_max"] <= bound).mean() * 100)
        short = (dec["actual_max"] - bound)
        short = short[short > 0]
        coverage[label] = {"covered_pct": covered,
                           "mean_shortfall_pp": float(short.mean()) if len(short) else 0.0,
                           "worst_shortfall_pp": float(short.max()) if len(short) else 0.0}
        print(f"  {label:36s} {covered:5.0f}% of {len(dec)} decisions"
              f"   mean shortfall {coverage[label]['mean_shortfall_pp']:.1f} pp"
              f"   worst {coverage[label]['worst_shortfall_pp']:.1f} pp")

    worst = dec.loc[dec["actual_max"].idxmax()]
    print(f"  the day's peak: actual {worst['actual_max']:.1f}% vs prophet upper "
          f"{worst['prophet_upper_max']:.1f}% at origin {worst['origin']}")

    out = {"dataset": args.dataset, "horizon_min": args.horizon_min,
           "refit_min": args.refit_min, "origins": len(dec),
           "methods": summary,
           "prophet_mae_by_horizon": {int(h): float(v) for h, v in by_h.items()},
           "decision_coverage": coverage,
           "decisions": decisions}
    path = results_dir() / f"rolling_{args.dataset}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    long.to_csv(results_dir() / f"rolling_{args.dataset}_points.csv", index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
