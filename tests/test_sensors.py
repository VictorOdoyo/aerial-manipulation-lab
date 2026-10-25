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


def test_sensor_named_streams_do_not_depend_on_position_sampling_schedule():
    a = m.Suite()
    b = m.Suite()
    a.position([0, 2])
    ra = a.imu(0, 0, [0, 0])
    rb = b.imu(0, 0, [0, 0])
    assert (
        np.array_equal(ra["specific_force_m_s2"], rb["specific_force_m_s2"])
        and ra["gyro_rad_s"] == rb["gyro_rad_s"]
    )


def test_zero_noise_suite_exposes_declared_imu_bias_without_hidden_truth_correction():
    suite = m.Suite(gyro_std=0, accel_std=0)
    r = suite.imu(0, 0, [0, 0])
    assert np.allclose(r["specific_force_m_s2"], [0.03, 9.80665 - 0.02])
    assert r["gyro_rad_s"] == 0.005
    clipped = suite.imu(0, 0, [1000, 0])
    assert not clipped["valid"] and max(abs(clipped["specific_force_m_s2"])) <= 16 * 9.80665


def test_zero_noise_tachometer_reports_supplied_rotor_state_without_touching_imu_streams():
    suite = m.Suite()
    speed = np.array([700.0, 800.0])
    assert np.array_equal(m.rotor_tachometer(suite, speed, 0), speed)
