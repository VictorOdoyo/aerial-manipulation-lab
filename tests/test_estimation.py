import numpy as np
import pytest

from aeriallab import estimation as m


def test_attitude_prediction_integrates_measured_gyro_and_preserves_psd_covariance():
    r = m.Attitude()
    r.predict(0.2, 0.5)
    assert r.state[0] == pytest.approx(0.1) and np.linalg.eigvalsh(r.covariance).min() > 0
