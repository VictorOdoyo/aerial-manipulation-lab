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
