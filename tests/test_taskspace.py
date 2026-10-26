import numpy as np

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
