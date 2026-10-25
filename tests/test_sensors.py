import numpy as np
import pytest

from aeriallab import sensors as m


def test_accelerometer_reads_weight_support_at_rest_and_zero_in_free_fall():
    assert np.allclose(m.specific_force([0, 0], 0), [0, 9.80665])
    assert np.allclose(m.specific_force([0, -9.80665], 0.4), 0)


def test_joint_encoder_roundtrip_error_is_bounded_by_half_count():
    q = np.array([0.2, -0.7])
    r = m.encoder(q)
    assert np.max(abs(r["angle_rad"] - q)) <= r["lsb_rad"] / 2
    assert np.all(m.encoder([0, 0])["codes"] == 0)


def test_encoder_rate_filter_handles_wrap_without_false_impulse():
    r = m.EncoderVelocity(0.001)
    r.update([np.pi - 0.01, 0], 0.01)
    v = r.update([-np.pi + 0.01, 0], 0.01)
    assert v[0] == pytest.approx(2, rel=1e-4) and v[1] == 0
