import numpy as np

from aeriallab import trajectory as m


def test_quintic_reference_has_zero_endpoint_velocity_and_acceleration():
    r = m.quintic(np.array([0.0, 1.0, 2.0]), [0, 2], [1, 3], 2)
    assert np.allclose(r["position"][[0, -1]], [[0, 2], [1, 3]])
    assert np.allclose(r["velocity"][[0, -1]], 0) and np.allclose(r["acceleration"][[0, -1]], 0)
    assert np.allclose(r["position"][1], [0.5, 2.5])
