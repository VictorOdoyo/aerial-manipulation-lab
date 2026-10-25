import numpy as np

from aeriallab import payload as m


def test_pickup_preserves_external_momentum_and_dissipates_capture_energy():
    from aeriallab.planar import Model, linear_momentum

    p = Model(gravity_m_s2=0)
    x = np.r_[[0.0, 2.0, 0.2, 0.3, -0.4], [1.0, 0.0, 0.1, 0.2, -0.1]]
    mass = 0.2
    velocity = np.array([0.0, 1.0])
    before = linear_momentum(x[:5], x[5:], p) + mass * velocity
    r = m.pickup(x, mass, velocity, p)
    assert np.allclose(linear_momentum(r["state"][:5], r["state"][5:], r["model"]), before)
    assert r["dissipated_j"] >= 0
