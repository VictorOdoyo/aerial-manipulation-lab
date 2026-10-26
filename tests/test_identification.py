import numpy as np

from aeriallab import identification as m


def test_payload_dynamics_are_affine():
    from dataclasses import replace

    from aeriallab.planar import Model, coriolis_matrix, gravity, mass_matrix

    q = np.array([0, 1, 0.1, 0.3, -0.5])
    v = np.array([0.2, 0, 0.1, 0.4, -0.2])
    a = np.array([0.1, 0.2, -0.1, 0.3, 0.5])
    r = m.payload_regressor(q, v, a)
    p = replace(Model(), payload_kg=0.37)
    actual = mass_matrix(q, p) @ a + coriolis_matrix(q, v, p) @ v + gravity(q, p)
    assert np.allclose(actual, r["baseline"] + 0.37 * r["column"])
