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
