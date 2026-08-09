"""Tests for the pure scaling logic -- runs without Prophet, pandas or numpy."""
import pytest

from scaling import ScalingDecision, decide, recommend, required_replicas


# --- the starter's arithmetic, kept working -------------------------------

def test_scale_up_when_cpu_exceeds_target():
    # 5 pods predicted at 85% CPU, target 60% -> ceil(5 * 85/60) = 8
    assert required_replicas(5, 85, target_cpu_per_replica=60) == 8


def test_clamps_to_minimum():
    assert required_replicas(5, 10, target_cpu_per_replica=60, min_replicas=2) == 2


def test_clamps_to_maximum():
    assert required_replicas(5, 99, target_cpu_per_replica=10, max_replicas=20) == 20


def test_no_change_when_at_target():
    assert required_replicas(5, 60, target_cpu_per_replica=60) == 5


def test_ceil_never_rounds_down_into_a_deficit():
    # 5 * 61/60 = 5.083 pods. Rounding down would leave every pod above target.
    assert required_replicas(5, 61, target_cpu_per_replica=60) == 6


def test_decision_action_text():
    d = decide(5, 85, target_cpu_per_replica=60)
    assert isinstance(d, ScalingDecision)
    assert d.delta == 3 and "Scale UP by 3" in d.action


def test_decision_scale_down():
    d = decide(10, 30, target_cpu_per_replica=60)  # ceil(10*30/60) = 5
    assert d.recommended_replicas == 5 and "Scale DOWN by 5" in d.action


@pytest.mark.parametrize("bad", [
    dict(current_replicas=0, predicted_cpu=50),
    dict(current_replicas=5, predicted_cpu=-1),
    dict(current_replicas=5, predicted_cpu=50, target_cpu_per_replica=0),
    dict(current_replicas=5, predicted_cpu=50, min_replicas=10, max_replicas=4),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        required_replicas(**bad)


# --- the reactive floor ---------------------------------------------------

def test_floor_rescues_an_unforecast_spike():
    """The scenario the lab write-up asks about: the model says 45%, the box is
    at 95%. The floor must win, or the service stays under-provisioned."""
    d = recommend(4, predicted_cpu=45, observed_cpu=95, target_cpu_per_replica=60)
    assert d.recommended_replicas == 7      # ceil(4 * 95/60)
    assert d.driver == "observed"


def test_floor_is_silent_while_the_forecast_leads():
    """Load rising as predicted: the floor must not add a single pod, or the
    whole point of forecasting -- scaling *before* the load -- is lost."""
    d = recommend(4, predicted_cpu=80, observed_cpu=50, target_cpu_per_replica=60)
    assert d.recommended_replicas == 6      # ceil(4 * 80/60), the forecast
    assert d.driver == "forecast"


def test_floor_never_scales_down():
    """observed < predicted must not drag the count below the forecast, even
    when the current reading is near zero."""
    d = recommend(6, predicted_cpu=60, observed_cpu=1, target_cpu_per_replica=60)
    assert d.recommended_replicas == 6


def test_recommend_without_an_observation_matches_decide():
    a = recommend(5, predicted_cpu=85, target_cpu_per_replica=60)
    b = decide(5, 85, target_cpu_per_replica=60)
    assert a.recommended_replicas == b.recommended_replicas


def test_clamp_is_reported_as_the_driver():
    d = recommend(5, predicted_cpu=999, observed_cpu=999,
                  target_cpu_per_replica=60, max_replicas=20)
    assert d.recommended_replicas == 20 and d.driver == "clamp"
