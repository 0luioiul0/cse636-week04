"""Stretch goal: publish the forecast as a Prometheus metric for KEDA to scale on.

    python scripts/exporter.py --once          # emit once, scrape ourselves, exit
    python scripts/exporter.py                 # serve forever, refresh every 5 min

The metric is `predicted_cpu_next_30m{service=...}` -- the highest `yhat_upper`
Prophet expects in the next 30 minutes. `k8s/keda-scaledobject.yaml` turns it
into a replica count.

Two deliberate choices:

* The model is retrained on a timer, not on every scrape. Prometheus scrapes
  every 15-30 s; refitting that often would burn CPU to produce a number that
  changes on the hour.
* `predicted_cpu_next_30m` and `observed_cpu` are exported *separately* rather
  than pre-combined. Keeping the reactive floor in the KEDA trigger (as a second
  trigger -- KEDA takes the maximum) means a dead exporter degrades to plain
  reactive autoscaling instead of to no autoscaling.
"""
from __future__ import annotations

import argparse
import time
import warnings

import pandas as pd
from prometheus_client import Gauge, start_http_server

from dataio import as_prophet, load, quiet
from forecast_model import fit

warnings.filterwarnings("ignore")

predicted = Gauge("predicted_cpu_next_30m",
                  "Prophet-predicted max CPU % (upper 80% bound) for the next 30 min",
                  ["service"])
observed = Gauge("observed_cpu_current",
                 "Most recent observed fleet CPU %", ["service"])
model_age = Gauge("forecast_model_age_seconds",
                  "Seconds since the forecasting model was last refitted", ["service"])


def compute(series: pd.DataFrame, now: pd.Timestamp, horizon_min: int, weekly: bool):
    history = series[series["ds"] <= now]
    m = fit(history, weekly)
    stamps = pd.date_range(now + pd.Timedelta(minutes=5),
                           now + pd.Timedelta(minutes=horizon_min), freq="5min")
    fc = m.predict(pd.DataFrame({"ds": stamps}))
    return float(fc["yhat_upper"].max()), float(history["y"].iloc[-1])


def main() -> None:
    quiet()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--service", default="checkout-api")
    ap.add_argument("--horizon-min", type=int, default=30)
    ap.add_argument("--refit-s", type=int, default=300)
    ap.add_argument("--once", action="store_true",
                    help="emit one sample, scrape our own endpoint, exit")
    args = ap.parse_args()

    series = as_prophet(load("alibaba"))
    # Stand at the end of the trace and pretend it is now.
    now = series["ds"].max()

    start_http_server(args.port)
    print(f"serving /metrics on :{args.port}")

    while True:
        t0 = time.time()
        peak, last = compute(series, now, args.horizon_min, weekly=True)
        predicted.labels(service=args.service).set(peak)
        observed.labels(service=args.service).set(last)
        model_age.labels(service=args.service).set(0)
        print(f"predicted_cpu_next_30m = {peak:.1f}%   observed = {last:.1f}%   "
              f"(fit took {time.time()-t0:.1f}s)")

        if args.once:
            import urllib.request
            body = urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/metrics").read().decode()
            print("\n--- scraped from our own /metrics ---")
            for line in body.splitlines():
                if line.startswith(("predicted_cpu", "observed_cpu", "forecast_model")):
                    print(line)
            return

        time.sleep(args.refit_s)


if __name__ == "__main__":
    main()
