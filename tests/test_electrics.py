import pytest

from aeriallab import electrics as m


def test_induced_power_scales_as_thrust_to_three_halves():
    a = m.propulsion_power([10, 10], [0, 0], profile_coefficient=0)["shaft_w"]
    b = m.propulsion_power([20, 20], [0, 0], profile_coefficient=0)["shaft_w"]
    assert b / a == pytest.approx(2**1.5)
