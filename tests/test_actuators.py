import numpy as np

from aeriallab import actuators as m


def test_rotor_pair_wrench_has_signed_differential_torque():
    assert np.allclose(m.wrench([10, 12]), [22, 0.5])


def test_allocator_closes_feasible_wrench_and_retains_failure_residual():
    a = m.allocate(22, 0.5)
    assert np.allclose(a["thrusts_n"], [10, 12], atol=1e-8)
    b = m.allocate(30, 0, maximum_n=(0, 20))
    assert b["limited"] and b["thrusts_n"][0] == 0
    assert b["achieved"][0] <= 20


def test_rotor_step_matches_held_speed_command_exponential():
    a = m.rotor_step(np.zeros(2), np.full(2, 20.0), 0.035)
    assert np.allclose(a["speed_rad_s"], 1000 * (1 - np.exp(-1)))
    assert np.all(a["thrust_n"] < 20)


def test_voltage_and_health_derating_change_thrust_capacity_consistently():
    a = m.speed_ceiling(11.1, health=[1, 0.25])
    assert np.allclose(a, [500, 250])


def test_joint_effort_obeys_amplitude_and_slew_constraints():
    a = m.joint_effort([10, -10], [0, 0], 0.01)
    assert np.allclose(a["torque_nm"], [0.4, -0.4]) and a["limited"]


def test_rotor_can_coast_above_a_reduced_speed_ceiling():
    a = m.rotor_step([800.0, 800.0], [20.0, 20.0], 0.035, maximum_rad_s=[0.0, 500.0])
    assert 0 < a["speed_rad_s"][0] < 800
    assert 500 < a["speed_rad_s"][1] < 800
