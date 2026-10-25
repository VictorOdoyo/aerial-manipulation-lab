import numpy as np

from aeriallab import core as m


def test_timeline_retains_requested_endpoint_and_bounded_final_step():
    a = m.timeline(0.017, 0.005)
    assert np.allclose(a, [0, 0.005, 0.01, 0.015, 0.017])
    assert a[-1] == 0.017 and np.all(np.diff(a) <= 0.005 + 1e-15)
