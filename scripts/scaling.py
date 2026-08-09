"""Forecast -> replica count. Pure, dependency-free, and unit-tested.

The first two functions are the lab's arithmetic, kept deliberately identical to
the course starter (`project/forecasting/scaling.py`) so the numbers line up.
`recommend()` is the addition this repository argues for: a predictive
recommendation with a *reactive floor*, so a spike the model failed to forecast
still gets capacity instead of being ignored because the forecast said 40%.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

TARGET_CPU = 60.0
MIN_REPLICAS = 2
MAX_REPLICAS = 50


@dataclass
class ScalingDecision:
    current_replicas: int
    recommended_replicas: int
    driver: str = "forecast"   # which signal set the number: forecast | observed | clamp

    @property
    def delta(self) -> int:
        return self.recommended_replicas - self.current_replicas

    @property
    def action(self) -> str:
        if self.delta > 0:
            return f"Scale UP by {self.delta} replica(s) (pre-emptive)"
        if self.delta < 0:
            return f"Scale DOWN by {-self.delta} replica(s)"
        return "No change needed"


def required_replicas(
    current_replicas: int,
    predicted_cpu: float,
    target_cpu_per_replica: float = TARGET_CPU,
    min_replicas: int = MIN_REPLICAS,
    max_replicas: int = MAX_REPLICAS,
) -> int:
    """Replicas needed to hold average CPU near the target.

    If `current_replicas` pods are collectively predicted to run at
    `predicted_cpu`% and each should sit at `target_cpu_per_replica`%, we need
    ceil(current x predicted / target) pods, clamped to [min, max]. This is the
    same ratio the Kubernetes HPA computes, with a forecast substituted for the
    current reading.

    >>> required_replicas(5, 85, target_cpu_per_replica=60)
    8
    >>> required_replicas(5, 20, target_cpu_per_replica=60)
    2
    """
    if current_replicas < 1:
        raise ValueError("current_replicas must be >= 1")
    if predicted_cpu < 0:
        raise ValueError("predicted_cpu must be >= 0")
    if target_cpu_per_replica <= 0:
        raise ValueError("target_cpu_per_replica must be > 0")
    if min_replicas > max_replicas:
        raise ValueError("min_replicas must be <= max_replicas")

    raw = math.ceil(current_replicas * predicted_cpu / target_cpu_per_replica)
    return max(min_replicas, min(max_replicas, raw))


def decide(
    current_replicas: int,
    predicted_cpu: float,
    target_cpu_per_replica: float = TARGET_CPU,
    min_replicas: int = MIN_REPLICAS,
    max_replicas: int = MAX_REPLICAS,
) -> ScalingDecision:
    """Bundle the replica calculation with a human-readable action."""
    rec = required_replicas(current_replicas, predicted_cpu,
                            target_cpu_per_replica, min_replicas, max_replicas)
    return ScalingDecision(current_replicas, rec)


def recommend(
    current_replicas: int,
    predicted_cpu: float,
    observed_cpu: float | None = None,
    target_cpu_per_replica: float = TARGET_CPU,
    min_replicas: int = MIN_REPLICAS,
    max_replicas: int = MAX_REPLICAS,
) -> ScalingDecision:
    """Predictive recommendation with a reactive floor.

    A forecast can only be wrong in two directions, and they are not
    symmetrical. Forecasting high wastes money; forecasting low drops requests.
    So the recommendation is the *larger* of

      * what the forecast says we will need, and
      * what the reading on the clock right now already says we need.

    The floor costs nothing while the forecast is right -- the reactive number
    is below the predictive one whenever load is rising as expected -- and it is
    the only thing standing between an unforecast spike and a saturated service.
    Note that it is a floor, not a full reactive controller: it never scales
    *down*, so it cannot fight the forecast.

    >>> recommend(4, predicted_cpu=45, observed_cpu=95).recommended_replicas
    7
    >>> recommend(4, predicted_cpu=45, observed_cpu=95).driver
    'observed'
    """
    signal, driver = predicted_cpu, "forecast"
    if observed_cpu is not None and observed_cpu > predicted_cpu:
        signal, driver = observed_cpu, "observed"

    unclamped = math.ceil(current_replicas * signal / target_cpu_per_replica)
    rec = required_replicas(current_replicas, signal, target_cpu_per_replica,
                            min_replicas, max_replicas)
    if rec != unclamped:
        driver = "clamp"
    return ScalingDecision(current_replicas, rec, driver)
