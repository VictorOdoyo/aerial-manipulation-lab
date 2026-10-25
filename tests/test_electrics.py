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


def test_battery_quadratic_point_closes_terminal_and_internal_power():
    r = m.battery_point(0.8, 400)
    assert r["bus_v"] * r["current_a"] == pytest.approx(400)
    assert r["ocv_v"] * r["current_a"] == pytest.approx(400 + r["heat_w"])
    with pytest.raises(ValueError):
        m.battery_point(0.8, 10000)


def test_flight_battery_coulomb_count_has_physical_capacity_units():
    assert m.battery_step(0.8, 5, 360, 5) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        m.battery_step(0.001, 100, 1, 5)
