import numpy as np

from aeriallab import taskspace as m


def test_kinetic_metric_and_tip():
    from aeriallab.planar import mass_matrix

    q = np.array([0, 1, 0, 0.3, -0.5])
    assert m.metric(mass_matrix(q)).shape == (5, 5)
    assert m.tip(q)[1].shape == (2, 5)
