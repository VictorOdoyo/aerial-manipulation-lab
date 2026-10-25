import numpy as np
import pytest

from aeriallab import quadrotor as m


def test_quadrotor_model_rejects_nonphysical_principal_inertia():
    with pytest.raises(ValueError):
        m.Model(inertia=(0.01, 0.01, 1.0))


def test_quad_allocation_has_four_independent_wrench_axes():
    a = m.allocation_matrix()
    assert np.linalg.matrix_rank(a) == 4
    assert np.allclose(a @ np.full(4, 5), [20, 0, 0, 0])


def test_full_quad_allocation_closes_hover_and_retains_failed_rotor_residual():
    r = m.allocate([20, 0, 0, 0])
    assert np.allclose(r["thrusts_n"], 5)
    failed = m.allocate([30, 0, 0, 0], maximum_n=[0, 10, 10, 10])
    assert failed["thrusts_n"][0] == 0 and np.linalg.norm(failed["residual"]) > 0.01


def test_full_quad_hover_matches_weight_without_fictitious_angular_acceleration():
    p = m.Model()
    a, w = m.acceleration([1, 0, 0, 0], [0, 0, 0], [0, 0, 0], np.full(4, p.mass_kg * p.gravity / 4), model=p)
    assert np.allclose(a, 0) and np.allclose(w, 0)


def test_spatial_integrator_matches_analytic_ballistic_translation():
    p = m.Model()
    x = np.r_[[0.0, 0.0, 10.0], [1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    dt = 0.02
    y = m.step(x, np.zeros(4), dt, model=p)
    assert np.allclose(y[:3], x[:3] + x[3:6] * dt + [0, 0, -0.5 * p.gravity * dt**2])
    assert np.linalg.norm(y[6:10]) == pytest.approx(1)


def test_torque_free_spatial_rotation_conserves_energy_and_inertial_momentum():
    p = m.Model(gravity=0)
    x = np.r_[np.zeros(6), [1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
    e = m.energy(x, p)
    h = m.angular_momentum(x, p)
    for _ in range(100):
        x = m.step(x, np.zeros(4), 0.002, model=p)
    assert m.energy(x, p) == pytest.approx(e, abs=1e-10)
    assert np.allclose(m.angular_momentum(x, p), h, atol=1e-10)
