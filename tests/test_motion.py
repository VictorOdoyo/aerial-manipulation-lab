import numpy as np
import pytest

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


def test_short_request_retimed():
    r = m.synchronized([0, 1], 0.01, [0, 0], [1, 2], 1, 2, 10)
    assert r["extended"] and r["duration_s"] > 1


def test_continuous_path_not_endpoint_only():
    r = m.segment_clearance([-1, 0], [1, 0], [[0, 0]], [0.2])
    assert r["collision"] and r["clearance_m"][0] == pytest.approx(-0.2)
    assert not m.segment_clearance([-1, 1], [1, 1], [[0, 0]], [0.2])["collision"]


def test_waypoint_continuity():
    r = m.waypoints([1 - 1e-7, 1, 1 + 1e-7], [[0], [1], [0.5]], [1, 2])
    assert np.max(abs(r["position"] - 1)) < 1e-10
    assert np.max(abs(r["velocity"])) < 1e-10
    assert np.max(abs(r["acceleration"])) < 1e-4


def test_jerk_cost_matches_quadrature():
    t = np.linspace(0, 2, 10001)
    r = m.quintic(t, 2, [0, 1], [1, 3])
    numerical = np.trapezoid(np.sum(r["jerk"] ** 2, axis=1), t)
    assert numerical == pytest.approx(m.jerk_cost([0, 1], [1, 3], 2), rel=1e-6)
