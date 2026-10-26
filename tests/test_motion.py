import numpy as np

from aeriallab import motion as m


def test_quintic_end_conditions():
    r = m.quintic([0, 1, 2], 1, [0, 1], [2, 3])
    assert np.allclose(r["position"], [[0, 1], [2, 3], [2, 3]])
    assert np.allclose(r["velocity"], 0)
    assert np.allclose(r["acceleration"], 0)
    assert np.allclose(r["jerk"][-1], 0)
