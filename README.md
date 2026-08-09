# CSE636 Week 4 — time-series forecasting for Kubernetes autoscaling

**Chi Zhang.** The lab and the assignment, run against a **real** production
trace rather than the synthetic one, with the synthetic series kept as a control
— which turned out to matter more than expected.

| deliverable | where |
|---|---|
| Lab notebook (outputs stored) | [`notebooks/forecast.ipynb`](notebooks/forecast.ipynb) |
| Lab write-up (the three questions) | [`lab_notes.md`](lab_notes.md) |
| Assignment report | [`docs/report.md`](docs/report.md) |
| The three required plots | [`figures/metrics_overview.png`](figures/metrics_overview.png), [`figures/cpu_forecast.png`](figures/cpu_forecast.png), [`figures/cpu_eval.png`](figures/cpu_eval.png) |
| Raw transcripts of every run | [`evidence/`](evidence/) |
| Stretch goal (Prometheus + KEDA) | [`scripts/exporter.py`](scripts/exporter.py), [`k8s/keda-scaledobject.yaml`](k8s/keda-scaledobject.yaml) |

```bash
python -m venv .venv && .venv/Scripts/activate     # or source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py        # ~2 min: regenerates every figure, number and transcript
```

`run_all.py` writes each stage's real stdout to `evidence/NN-stage.txt` with the
command echoed above it, so anything quoted in the report can be traced to a
transcript. Add `--fetch` to re-download the trace.

## The five results worth reading the repository for

**1. At the horizon that matters, Prophet loses to a moving average.**
Retraining at every 30-minute origin across the held-out day and forecasting 30
minutes ahead — what an autoscaler actually does — on the real trace:

| method | MAE (pp), real trace | MAE (pp), synthetic |
|---|---|---|
| ARIMA(2,1,2), trailing 24 h | **4.64** | 2.78 |
| Prophet + level adjustment | 4.90 | 2.46 |
| 30-minute moving average | 4.92 | 2.89 |
| persistence | 5.41 | 3.00 |
| **Prophet** | **5.70** | **2.34** |
| seasonal naive (t − 24 h) | 7.31 | 6.74 |

**2. The handout's synthetic data silently validates the method it is meant to
test.** Same code, same protocol: Prophet ranks *fifth of six* on the real trace
and *first* on the synthetic one — because that series is a sum of two sines
plus Gaussian noise, which is Prophet's own model class. Reporting only the
synthetic number would overstate the method by half.

**3. Prophet has no autoregressive term, and it shows.** `yhat(t)` is a
function of the clock, not of what CPU is doing now, so a fleet that sits above
its seasonal curve for an hour stays under-forecast. Adding back the mean
residual of the last 30 minutes — six lines — takes MAE from 5.70 to 4.90 and
peak coverage from 60% to 71%.

**4. Predictive autoscaling cost 12% more than reactive HPA and bought no
measurable risk reduction at 60-second pod startup.** Seven days of the trace
replayed through a simulated HPA (metric lag, tolerance, stabilisation window,
scale-up policy) and a Prophet controller:

| scenario | mean pods | $/week | vs static | pods started | min > 90% CPU |
|---|---|---|---|---|---|
| static over-provisioning | 31.0 | 208.32 | — | 0 | 0 |
| reactive HPA | 16.8 | **113.07** | −45.7% | 2 225 | 0 |
| predictive | 18.5 | 124.11 | −40.4% | 129 | 0 |
| predictive + reactive floor | 18.7 | 125.63 | −39.7% | 237 | 0 |
| floor at HPA cadence | 18.9 | 126.96 | −39.1% | 541 | 0 |

What it *did* buy: a quarter of the pod churn, and — once pod startup is raised
to 10 minutes — 1 minute above 90% CPU where the HPA spends 23.

**5. A purely predictive controller is worse than the HPA it replaced on
anything it did not predict.** Under an unforecastable flash crowd (×2 for 20
min) it dropped **4×** more load than plain reactive HPA and stayed wrong for the
full 20 minutes. The reactive floor in
[`scripts/scaling.py`](scripts/scaling.py) closes the gap — but only when it is
evaluated at the *controller's* cadence, not the forecaster's.

## The dataset, and the handout's dead link

The handout offers the Alibaba trace at
`github.com/alibaba/clusterdata/raw/master/cluster-trace-v2018/machine_usage.csv`,
which **404s** — that CSV is not in the repository. The real source is the OSS
bucket named in the trace's own `fetchData.sh`, where `machine_usage` is a
**1.77 GB** tarball wrapping a **9.0 GB** CSV (~246 M rows, 4 034 machines ×
8 days).

Rows are grouped by machine, so [`scripts/fetch_alibaba.py`](scripts/fetch_alibaba.py)
streams the tarball, keeps the first 24 machine ids and hangs up: **2 MB
transferred** instead of 1.77 GB. [`scripts/prepare_data.py`](scripts/prepare_data.py)
resamples each machine onto a 5-minute grid and averages the 22 serving machines
(an HPA reads the mean across pods, not one pod). Two machines idle at ~5% CPU
and are excluded, with the exclusion printed.

What is in it: **2 304 five-minute buckets, 8.00 days**, mean CPU 40.4%, range
16.5–82.7%, daily-lag autocorrelation **+0.68**, a sharp morning ramp — and a
**5.2-hour window in which not one machine reported**. That gap is left as a gap
for the model (Prophet regresses on time and needs no value there); it is
interpolated only inside the simulator, which needs a number every tick.

## Repository map

```
scripts/
  fetch_alibaba.py     stream 1.77 GB, keep 2 MB
  prepare_data.py      -> data/alibaba_5min.csv, data/synthetic_5min.csv
  dataio.py            one definition of the train/test split, used everywhere
  explore.py           -> figures/metrics_overview.png + the gap report
  forecast_model.py    Prophet fit/eval -> the three required plots
  rolling_eval.py      30-min horizon, refit per origin, 5 baselines
  reproducibility.py   12 identical runs -> how much yhat_upper wobbles
  scaling.py           PURE forecast -> replicas, with the reactive floor
  simulate.py          HPA algorithm + 5 controllers over 7 days -> cost & risk
  sweep.py             pod-startup and flash-crowd sweeps
  exporter.py          Prometheus gauge for KEDA (stretch goal)
  run_all.py           regenerate everything into evidence/
  tests/               17 tests on the pure scaling logic
k8s/keda-scaledobject.yaml   two triggers: forecast + reactive floor
figures/  results/  evidence/  data/  notebooks/  docs/
```

## Things that are true and easy to get wrong

- **The HPA does not average over ready pods and multiply by the total.** My
  first simulator did, which double-counts starting capacity and produced
  60-replica overshoots real Kubernetes does not have. The documented algorithm
  excludes unready pods from the sum and assumes they consume 0% when scaling
  up; that is the damping that keeps the loop stable. Fixed, and the fix is why
  the reactive trace in the figures looks the way it does.
- **`yhat` is deterministic; `yhat_upper` is not.** It is a Monte-Carlo estimate
  from 1 000 unseeded draws. Twelve identical runs spread it over 0.91 points and
  flipped the recommendation between 17 and 18 replicas
  ([`evidence/04-reproducibility.txt`](evidence/04-reproducibility.txt)).
- **The 80% interval is not 80% here.** It covered 63.7% of the held-out day,
  and `max(yhat_upper)` covered the actual 30-minute peak in 60% of decisions.
  The "conservative estimate" is less conservative than its name.
- **The handout's synthetic series never crosses its own 70% scale-up
  threshold** (peak 69.7%), so the threshold drawn on its plots never fires.
- **`freq="5T"` is deprecated** in pandas ≥ 2.2; this repository uses `"5min"`.
- **Memory is not a scaling signal on this fleet.** It sits at ~90% and barely
  moves.

## AI disclosure

Claude (Opus 5) in Claude Code was used to write the scripts and draft the
report. Every number is produced by code in this repository and captured in
`evidence/`; the model's claims were not taken on trust — the "Prophet beats
simple baselines" assumption it started from is false on this data, which is what
`rolling_eval.py` exists to show. Full statement at the end of
[`docs/report.md`](docs/report.md).
