import numpy as np

from aeriallab import environment as m


def test_aerodynamic_drag_opposes_relative_motion_and_dissipates_energy():
    a = m.drag([2, -1], wind=[0, 0])
    assert a["force_n"][0] < 0 and a["force_n"][1] > 0
    assert a["relative_dissipation_w"] > 0


def test_wind_gust_is_zero_at_boundaries_and_reaches_declared_peak():
    assert np.allclose(m.gust([1, 2, 3, 4, 5]), [0, 0, 5, 0, 0], atol=1e-12)
