import numpy as np

from aeriallab import studies_taskspace as m


def test_inertia():
    s = m.inertia()["summary"]
    assert s["loaded_effective_mass_kg"] > s["minimum_effective_mass_kg"]


def test_velocity():
    r = m.velocity()
    assert np.all(r["arrays"]["dynamic_j"] <= r["arrays"]["euclidean_j"] + 1e-12)


def test_nullspace():
    s = m.nullspace()["summary"]
    assert s["maximum_tip_leak_m_s"] < 1e-10 and s["minimum_barrier_descent"] > 0


def test_constraint():
    assert m.constraint()["summary"]["max_constraint_residual_m_s2"] < 1e-9


def test_work():
    assert m.work()["summary"]["max_power_defect_w"] < 1e-10
