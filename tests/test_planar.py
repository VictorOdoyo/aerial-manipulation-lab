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
