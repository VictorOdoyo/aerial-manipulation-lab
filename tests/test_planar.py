import numpy as np
import pytest

from aeriallab import planar as m


def test_planar_model_has_positive_body_and_link_properties():
    p = m.Model()
    assert p.total_mass == pytest.approx(2.33)
    with pytest.raises(ValueError):
        m.Model(link1_m=-1)


def test_tip_jacobian_matches_independent_centered_position_difference():
    q = np.array([1.0, 2.0, 0.2, 0.3, -0.4])
    a = m.kinematics(q)
    h = 1e-6
    numerical = np.column_stack(
        [
            (m.kinematics(q + np.eye(5)[k] * h)["tip"][0] - m.kinematics(q - np.eye(5)[k] * h)["tip"][0])
            / (2 * h)
            for k in range(5)
        ]
    )
    assert np.allclose(a["tip"][1], numerical, atol=1e-9)


def test_coupled_mass_matrix_is_spd_and_includes_base_arm_cross_terms():
    matrix = m.mass_matrix([0, 2, 0.2, 0.3, -0.4])
    assert np.allclose(matrix, matrix.T) and np.linalg.eigvalsh(matrix).min() > 0
    assert abs(matrix[0, 3]) > 0.01 and abs(matrix[2, 4]) > 0.001


def test_gravity_is_independent_potential_gradient_and_total_weight():
    q = np.array([0.0, 2.0, 0.2, 0.4, -0.3])
    g = m.gravity(q)
    h = 1e-6
    numerical = np.array(
        [(m.potential(q + np.eye(5)[k] * h) - m.potential(q - np.eye(5)[k] * h)) / (2 * h) for k in range(5)]
    )
    assert np.allclose(g, numerical, atol=1e-8)
    assert g[1] == pytest.approx(m.Model().total_mass * 9.80665)


def test_mass_complex_step_derivatives_match_centered_real_differences():
    q = np.array([0.0, 2.0, 0.2, 0.4, -0.3])
    d = m.mass_derivatives(q)
    h = 1e-5
    numerical = np.stack(
        [
            (m.mass_matrix(q + np.eye(5)[k] * h) - m.mass_matrix(q - np.eye(5)[k] * h)) / (2 * h)
            for k in range(5)
        ],
        axis=2,
    )
    assert np.allclose(d, numerical, atol=1e-9)


def test_coriolis_preserves_mass_derivative_skew_identity():
    q = np.array([0.0, 2.0, 0.2, 0.4, -0.3])
    v = np.array([1.0, -0.2, 0.5, 0.3, -0.4])
    mdot = np.einsum("ijk,k->ij", m.mass_derivatives(q), v)
    c = m.coriolis_matrix(q, v)
    skew = mdot - 2 * c
    assert np.allclose(skew + skew.T, 0, atol=1e-12)


def test_internal_joint_torques_do_not_invent_external_base_wrench():
    q = [0, 2, 0.2, 0.3, -0.2]
    a = m.generalized_force(q, 0, joint_torques=[1, 2])
    assert np.allclose(a, [0, 0, 0, 1, 2])
    assert np.allclose(m.generalized_force(q, 10)[:2], [-10 * np.sin(0.2), 10 * np.cos(0.2)])


def test_free_fall_translates_whole_articulated_system_at_gravity():
    q = [0, 2, 0.2, 0.3, -0.4]
    a = m.acceleration(q, np.zeros(5), np.zeros(5))
    assert np.allclose(a, [0, -9.80665, 0, 0, 0], atol=1e-12)


def test_coupled_rk4_matches_independent_adaptive_integrator():
    from scipy.integrate import solve_ivp

    x = np.r_[[0.0, 2.0, 0.2, 0.4, -0.3], [0.1, -0.2, 0.3, 0.4, -0.2]]
    force = np.array([0.0, 23.0, 0.1, 0.05, -0.03])
    dt = 0.005
    reference = solve_ivp(lambda _t, y: m.rhs(y, force), (0, dt), x, rtol=1e-11, atol=1e-12).y[:, -1]
    assert np.allclose(m.step(x, force, dt), reference, atol=1e-8)
