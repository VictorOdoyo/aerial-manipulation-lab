import numpy as np
import pytest

from aeriallab import estimation as m


def test_attitude_prediction_integrates_measured_gyro_and_preserves_psd_covariance():
    r = m.Attitude()
    r.predict(0.2, 0.5)
    assert r.state[0] == pytest.approx(0.1) and np.linalg.eigvalsh(r.covariance).min() > 0


def test_attitude_filter_estimates_constant_gyro_bias_from_reference_angles():
    r = m.Attitude()
    for _ in range(500):
        r.predict(0.01, 0.01)
        m.correct_attitude(r, 0)
    assert abs(r.state[0]) < 0.001 and abs(r.state[1] - 0.01) < 0.001
    assert not m.correct_attitude(r, 2)["accepted"]


def test_navigation_prediction_matches_analytic_constant_acceleration_and_bias():
    x = np.zeros(6)
    x[4:] = [0.1, -0.2]
    state, p = m.navigation_predict(x, np.eye(6), [1, 2], 0.5)
    assert np.allclose(state[:2], 0.5 * np.array([0.9, 2.2]) * 0.5**2)
    assert np.allclose(state[2:4], np.array([0.9, 2.2]) * 0.5)
    assert np.linalg.eigvalsh(p).min() > 0


def test_navigation_update_gates_outlier_and_reduces_position_uncertainty():
    x, p, accepted, _nis = m.navigation_correct(np.zeros(6), np.eye(6), [0.1, 0.2])
    assert accepted and p[0, 0] < 0.001 and x[0] > 0.09
    assert not m.navigation_correct(np.zeros(6), np.eye(6), [100, 100])[2]


def test_navigation_observer_uses_supplied_sensors_and_gates_position_outlier():
    r = m.Navigation([0, 2])
    imu = {"gyro_rad_s": 0.0, "specific_force_m_s2": np.array([0.0, 9.80665]), "valid": True}
    for _ in range(100):
        r.update(imu, 0.01, position=[0, 2], attitude_reference=0)
    assert np.linalg.norm(r.state[:4] - [0, 2, 0, 0]) < 1e-8
    result = r.update(imu, 0.01, position=[100, 100])
    assert not result["position_accepted"]
