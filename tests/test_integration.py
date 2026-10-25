import numpy as np

from aeriallab import integration as m


def test_body_fixed_wrench_integration_closes_work_energy():
    from aeriallab.planar import Model

    p = Model()
    x = np.r_[[0.0, 2.0, 0.2, 0.3, -0.4], [0.1, -0.2, 0.3, 0.4, -0.2]]
    a = m.wrench_step(x, 23, 0.1, [0.05, -0.03], 0.002, p)
    assert abs(a["mechanical_defect_j"]) < 1e-7
