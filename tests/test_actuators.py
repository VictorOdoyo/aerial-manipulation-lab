import numpy as np

from aeriallab import actuators as m


def test_rotor_pair_wrench_has_signed_differential_torque():
    assert np.allclose(m.wrench([10, 12]), [22, 0.5])
