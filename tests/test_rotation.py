import numpy as np

from aeriallab import rotation as m


def test_quaternion_composition_respects_identity_and_inverse():
    q = m.normalize([1, 2, 3, 4])
    conjugate = q * np.array([1, -1, -1, -1])
    assert np.allclose(m.multiply(q, conjugate), [1, 0, 0, 0])
