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
