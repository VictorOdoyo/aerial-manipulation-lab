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
