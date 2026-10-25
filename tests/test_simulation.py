import numpy as np
import pytest

from aeriallab import simulation as m


def test_replay_config_rejects_ambiguous_switches_and_unresolved_timing():
    with pytest.raises(ValueError):
        m.Config(dt_s=0.1)
    with pytest.raises(ValueError):
        m.Config(moving_arm=1)
    with pytest.raises(ValueError):
        m.Config(position_outage=(3, 2))
    assert m.Config().feedback == "estimated"


def test_causal_short_flight_replay_remains_finite_and_uses_real_sensor_updates():
    r = m.run(m.Config(duration_s=0.3))
    a = r["arrays"]
    assert r["termination"] == "completed"
    assert np.isfinite(a["state"]).all() and np.min(a["state"][:, 1]) > 1.9
    assert a["position_available"].sum() == 3
    assert a["battery_soc"][-1] < a["battery_soc"][0]
    assert np.max(abs(a["mechanical_defect_j"])) < 1e-4


def test_position_outlier_is_rejected_without_replacing_the_measurement_with_truth():
    r = m.run(m.Config(duration_s=0.3, position_outlier=True, outlier_time_s=0.1))
    a = r["arrays"]
    assert a["position_available"].sum() == 3
    assert np.sum(a["position_available"] & ~a["position_accepted"]) >= 1
    assert max(a["position_nis"]) > 9.21034


def test_nondivisible_flight_duration_is_actually_completed():
    r = m.run(m.Config(duration_s=0.017))
    assert r["arrays"]["time_s"][-1] == 0.017 and r["termination"] == "completed"


@pytest.mark.parametrize("value", [None, [], {"unknown": 1}, {"dt_s": 0.5}, {"moving_arm": 1}])
def test_custom_flight_config_rejects_ambiguous_input(value):
    with pytest.raises(ValueError):
        m.Config.from_dict(value)
