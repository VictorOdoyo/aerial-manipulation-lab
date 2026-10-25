import numpy as np
import pytest

from aeriallab import contact as m


def test_plane_contact_is_unilateral_and_absent_above_surface():
    q = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    v = np.zeros(5)
    assert np.allclose(m.plane_force(q, v, height_m=0)["force_n"], 0)
    r = m.plane_force(q, v, height_m=0.5)
    assert r["force_n"][1] > 0 and r["penetration_m"] > 0


def test_contact_impulse_enforces_restitution_without_energy_creation():
    from aeriallab.planar import tip_velocity

    q = np.array([0.0, 1.0, 0.2, 0.3, -0.4])
    v = np.array([1.0, -2.0, 0.3, 0.4, -0.2])
    before = tip_velocity(q, v)[1]
    r = m.normal_impulse(q, v, 0.3)
    assert tip_velocity(q, r["velocity"])[1] == pytest.approx(-0.3 * before)
    assert r["dissipated_j"] > 0


def test_joint_stop_reacts_on_floating_base_and_removes_stopped_velocity():
    q = [0, 2, 0.2, 0.3, -0.4]
    v = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    r = m.joint_stop(q, v)
    assert abs(r["velocity"][3]) < 1e-12 and abs(r["velocity"][2]) > 1e-3
    assert r["dissipated_j"] > 0
