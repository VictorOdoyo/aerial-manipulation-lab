import numpy as np
import pytest

from aeriallab import identification as m


def test_payload_dynamics_are_affine():
    from dataclasses import replace

    from aeriallab.planar import Model, coriolis_matrix, gravity, mass_matrix

    q = np.array([0, 1, 0.1, 0.3, -0.5])
    v = np.array([0.2, 0, 0.1, 0.4, -0.2])
    a = np.array([0.1, 0.2, -0.1, 0.3, 0.5])
    r = m.payload_regressor(q, v, a)
    p = replace(Model(), payload_kg=0.37)
    actual = mass_matrix(q, p) @ a + coriolis_matrix(q, v, p) @ v + gravity(q, p)
    assert np.allclose(actual, r["baseline"] + 0.37 * r["column"])


def test_payload_fit_known_noise():
    phi = np.linspace(0.2, 2, 30)
    r = m.payload_fit(phi, 0.4 * phi, 0.01)
    assert r["payload_kg"] == pytest.approx(0.4)
    assert r["standard_deviation_kg"] < 0.01
    with pytest.raises(ValueError):
        m.payload_fit(np.zeros(30), np.zeros(30), 0.01)


def test_damping_recovery():
    v = np.random.default_rng(4).normal(size=(100, 5))
    truth = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.allclose(m.damping_fit(v, v * truth)["damping"], truth)


def test_contact_wrench_inverse():
    from aeriallab.planar import kinematics

    jac = kinematics([0, 1, 0, 0.3, -0.5])["tip"][1]
    force = np.array([2.0, 4.0])
    r = m.wrench_fit(jac, jac.T @ force, 0.01)
    assert np.allclose(r["force"], force)


def test_friction_cone_projection():
    r = m.friction_projection([4, 1], 0.5)
    assert abs(r[0]) <= 0.5 * r[1] + 1e-12 and r[1] >= 0
    assert np.allclose(m.friction_projection([0, -2]), 0)
    assert np.allclose(m.friction_projection([0.1, 2]), [0.1, 2])
