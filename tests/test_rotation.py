import numpy as np
import pytest

from aeriallab import rotation as m


def test_quaternion_composition_respects_identity_and_inverse():
    q = m.normalize([1, 2, 3, 4])
    conjugate = q * np.array([1, -1, -1, -1])
    assert np.allclose(m.multiply(q, conjugate), [1, 0, 0, 0])


def test_rotation_matrix_preserves_norm_and_positive_orientation():
    r = m.matrix([1, 2, 3, 4])
    assert np.allclose(r.T @ r, np.eye(3)) and np.linalg.det(r) == pytest.approx(1)


def test_constant_body_rate_matches_known_axis_rotation():
    q = m.integrate([1, 0, 0, 0], [0, 0, np.pi], 0.5)
    assert np.allclose(m.matrix(q) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
