"""How repeatable is the number the autoscaler acts on?

Prophet's point forecast is a deterministic MAP fit, but `yhat_upper` -- the
value the scaling rule actually uses -- is estimated by drawing
`uncertainty_samples` (1000 by default) Monte-Carlo trajectories. Nothing seeds
that draw, so the safety margin wobbles from run to run even with identical
data and identical code.

This measures the wobble and translates it into replicas, because "the forecast
moved 0.4 points" only matters if it changes the decision.

    python scripts/reproducibility.py --runs 12
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from dataio import as_prophet, load, quiet, results_dir
from forecast_model import fit
from scaling import required_replicas

warnings.filterwarnings("ignore")


def main() -> None:
    quiet()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--current-replicas", type=int, default=20)
    ap.add_argument("--target-cpu", type=float, default=60.0)
    args = ap.parse_args()

    series = as_prophet(load("alibaba"))
    now = series["ds"].max()
    stamps = pd.date_range(now + pd.Timedelta(minutes=5),
                           now + pd.Timedelta(minutes=30), freq="5min")

    yhat, upper, replicas = [], [], []
    for i in range(args.runs):
        m = fit(series, weekly=True)
        fc = m.predict(pd.DataFrame({"ds": stamps}))
        yhat.append(float(fc["yhat"].max()))
        upper.append(float(fc["yhat_upper"].max()))
        replicas.append(required_replicas(args.current_replicas, upper[-1],
                                          args.target_cpu))
        print(f"  run {i+1:2d}: yhat_max {yhat[-1]:.3f}  yhat_upper_max "
              f"{upper[-1]:.3f}  -> {replicas[-1]} replicas")

    out = {
        "runs": args.runs,
        "yhat_max": {"mean": float(np.mean(yhat)), "std": float(np.std(yhat)),
                     "spread": float(np.ptp(yhat))},
        "yhat_upper_max": {"mean": float(np.mean(upper)), "std": float(np.std(upper)),
                           "spread": float(np.ptp(upper))},
        "replicas": {"min": int(min(replicas)), "max": int(max(replicas)),
                     "distinct": sorted(set(replicas))},
    }
    print(f"\nyhat_max       : {out['yhat_max']['mean']:.3f} "
          f"+/- {out['yhat_max']['std']:.3f} (spread {out['yhat_max']['spread']:.3f} pp)")
    print(f"yhat_upper_max : {out['yhat_upper_max']['mean']:.3f} "
          f"+/- {out['yhat_upper_max']['std']:.3f} "
          f"(spread {out['yhat_upper_max']['spread']:.3f} pp)")
    print(f"replica recommendation across {args.runs} identical runs: "
          f"{out['replicas']['distinct']}")

    path = results_dir() / "reproducibility.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
