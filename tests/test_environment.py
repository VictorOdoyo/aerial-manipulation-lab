import numpy as np
import pytest

from aeriallab import environment as m


def test_aerodynamic_drag_opposes_relative_motion_and_dissipates_energy():
    a = m.drag([2, -1], wind=[0, 0])
    assert a["force_n"][0] < 0 and a["force_n"][1] > 0
    assert a["relative_dissipation_w"] > 0


def test_wind_gust_is_zero_at_boundaries_and_reaches_declared_peak():
    assert np.allclose(m.gust([1, 2, 3, 4, 5]), [0, 0, 5, 0, 0], atol=1e-12)


def test_ground_effect_approaches_free_air_at_large_clearance():
    assert m.ground_effect(100) == pytest.approx(1, abs=1e-6)
    assert m.ground_effect(0.12) > 1
    with pytest.raises(ValueError):
        m.ground_effect(0.01)
