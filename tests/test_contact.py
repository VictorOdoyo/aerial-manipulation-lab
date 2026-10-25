import numpy as np

from aeriallab import contact as m


def test_plane_contact_is_unilateral_and_absent_above_surface():
    q = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    v = np.zeros(5)
    assert np.allclose(m.plane_force(q, v, height_m=0)["force_n"], 0)
    r = m.plane_force(q, v, height_m=0.5)
    assert r["force_n"][1] > 0 and r["penetration_m"] > 0
