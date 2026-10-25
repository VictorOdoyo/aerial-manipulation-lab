import numpy as np

from aeriallab import sensors as m


def test_accelerometer_reads_weight_support_at_rest_and_zero_in_free_fall():
    assert np.allclose(m.specific_force([0, 0], 0), [0, 9.80665])
    assert np.allclose(m.specific_force([0, -9.80665], 0.4), 0)
