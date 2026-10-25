import pytest

from aeriallab import electrics as m


def test_induced_power_scales_as_thrust_to_three_halves():
    a = m.propulsion_power([10, 10], [0, 0], profile_coefficient=0)["shaft_w"]
    b = m.propulsion_power([20, 20], [0, 0], profile_coefficient=0)["shaft_w"]
    assert b / a == pytest.approx(2**1.5)


def test_servo_braking_dissipates_mechanical_energy_without_claimed_regeneration():
    r = m.servo_power([1, -2], [2, 3])
    assert r["mechanical_w"] == -4
    assert r["electrical_w"] - r["heat_w"] == pytest.approx(-4)
    assert r["electrical_w"] > 0
