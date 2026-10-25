import numpy as np

from aeriallab import trajectory as m


def test_quintic_reference_has_zero_endpoint_velocity_and_acceleration():
    r = m.quintic(np.array([0.0, 1.0, 2.0]), [0, 2], [1, 3], 2)
    assert np.allclose(r["position"][[0, -1]], [[0, 2], [1, 3]])
    assert np.allclose(r["velocity"][[0, -1]], 0) and np.allclose(r["acceleration"][[0, -1]], 0)
    assert np.allclose(r["position"][1], [0.5, 2.5])


def test_waypoint_reference_is_continuous_at_segment_boundary():
    p = [[0, 2], [1, 3], [2, 2]]
    a = m.waypoints(1.0, [0, 1, 2], p)
    assert np.allclose(a["position"], [1, 3]) and np.allclose(a["velocity"], 0)
    assert np.allclose(m.waypoints(10, [0, 1, 2], p)["position"], p[-1])
