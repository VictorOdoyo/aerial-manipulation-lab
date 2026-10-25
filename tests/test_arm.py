import numpy as np
import pytest

from aeriallab import arm as m


def test_inverse_kinematics_recovers_forward_endpoint_and_rejects_unreachable_pose():
    from aeriallab.planar import kinematics

    q = np.array([1.0, 2.0, 0.2, 0.4, -0.5])
    target = kinematics(q)["tip"][0]
    joints = m.inverse_kinematics(q[:3], target, -1)
    assert np.allclose(joints, q[3:], atol=1e-12)
    with pytest.raises(ValueError):
        m.inverse_kinematics([0, 2, 0], [10, 2])
