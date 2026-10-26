import numpy as np
import pytest

from aeriallab import taskspace as m


def test_kinetic_metric_and_tip():
    from aeriallab.planar import mass_matrix

    q = np.array([0, 1, 0, 0.3, -0.5])
    assert m.metric(mass_matrix(q)).shape == (5, 5)
    assert m.tip(q)[1].shape == (2, 5)


def test_operational_mass_inverse():
    from aeriallab.planar import mass_matrix

    q = np.array([0, 1, 0, 0.3, -0.5])
    mass = mass_matrix(q)
    jac = m.tip(q)[1]
    lam = m.operational_inertia(mass, jac)
    assert np.allclose(lam @ jac @ np.linalg.solve(mass, jac.T), np.eye(2))
    assert min(np.linalg.eigvalsh(lam)) > 0


def test_dynamic_nullspace():
    from aeriallab.planar import mass_matrix

    q = np.array([0, 1, 0, 0.3, -0.5])
    mass = mass_matrix(q)
    jac = m.tip(q)[1]
    r = m.dynamic_inverse(mass, jac)
    assert np.allclose(jac @ r["null"], 0, atol=1e-12)
    assert np.allclose(r["null"] @ r["null"], r["null"])
    assert np.allclose(jac @ np.linalg.solve(mass, r["null"].T), 0, atol=1e-12)


def test_minimum_energy_vs_null_motion():
    from aeriallab.planar import mass_matrix

    q = np.array([0, 1, 0, 0.3, -0.5])
    mass = mass_matrix(q)
    jac = m.tip(q)[1]
    r = m.minimum_energy_velocity(mass, jac, [0.2, -0.1])
    extra = m.dynamic_inverse(mass, jac)["null"] @ np.ones(5)
    shifted = r["generalized_velocity"] + extra
    assert np.allclose(jac @ shifted, [0.2, -0.1])
    assert 0.5 * shifted @ mass @ shifted >= r["kinetic_j"]


def test_convective_second_derivative():
    q = np.array([0, 1, 0.1, 0.3, -0.5])
    v = np.array([0.2, -0.1, 0.3, 0.5, -0.2])
    h = 1e-4
    numeric = (m.tip(q + h * v)[0] - 2 * m.tip(q)[0] + m.tip(q - h * v)[0]) / h**2
    assert np.allclose(m.convective_acceleration(q, v), numeric, atol=1e-7)


def test_impedance_virtual_work():
    q = np.array([0, 1, 0.1, 0.3, -0.5])
    v = np.array([0.2, -0.1, 0.3, 0.5, -0.2])
    r = m.impedance(q, v, [0.2, 0.4])
    assert r["generalized_force"] @ v == pytest.approx(r["task_power_w"])


def test_task_force_limit_scaling():
    jac = m.tip([0, 1, 0, 0.4, -0.6])[1]
    r = m.scale_force(jac, [100, 20], [20, 20, 2, 1, 1])
    assert np.all(abs(r["generalized"]) <= np.array([20, 20, 2, 1, 1]) + 1e-12)
    assert r["scale"] < 1


def test_constraint_kkt_balance():
    from aeriallab.planar import gravity, mass_matrix

    q = np.array([0, 1, 0, 0.3, -0.5])
    mass = mass_matrix(q)
    jac = m.tip(q)[1]
    bias = gravity(q)
    r = m.constrained_acceleration(mass, jac, np.zeros(5), bias, np.zeros(2), np.zeros(2))
    assert np.allclose(jac @ r["acceleration"], 0, atol=1e-10)
    assert np.allclose(mass @ r["acceleration"], -bias + jac.T @ r["reaction"])


def test_com_gravity_gradient():
    from aeriallab.planar import Model, gravity

    q = np.array([0, 1, 0.2, 0.3, -0.5])
    p = Model()
    r = m.center_of_mass(q, p)
    assert np.allclose(gravity(q, p), p.total_mass * p.gravity_m_s2 * r["jacobian"][1])
