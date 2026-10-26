import numpy as np

from aeriallab import motion as m


def test_quintic_end_conditions():
    r = m.quintic([0, 1, 2], 1, [0, 1], [2, 3])
    assert np.allclose(r["position"], [[0, 1], [2, 3], [2, 3]])
    assert np.allclose(r["velocity"], 0)
    assert np.allclose(r["acceleration"], 0)
    assert np.allclose(r["jerk"][-1], 0)


def test_duration_respects_all_bounds():
    duration = m.minimum_duration([0, 0], [1, -2], [1, 2], [2, 3], [10, 20])
    r = m.quintic(np.linspace(0, duration, 10001), duration, [0, 0], [1, -2])
    assert np.all(np.max(abs(r["velocity"]), axis=0) <= np.array([1, 2]) + 1e-8)
    assert np.all(np.max(abs(r["acceleration"]), axis=0) <= np.array([2, 3]) + 1e-8)
    assert np.all(np.max(abs(r["jerk"]), axis=0) <= np.array([10, 20]) + 1e-8)
