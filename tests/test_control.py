import numpy as np

from aeriallab import control as m


def test_hover_trim_balances_extended_arm_and_exposes_impossible_tilted_hover():
    from aeriallab.planar import acceleration, generalized_force

    q = np.array([0.0, 2.0, 0.0, 0.5, -0.6])
    r = m.hover_trim(q)
    force = generalized_force(q, r["thrust_n"], r["body_torque_nm"], r["joint_torques_nm"])
    assert r["feasible"] and np.allclose(acceleration(q, np.zeros(5), force), 0, atol=1e-12)
    assert not m.hover_trim([0, 2, 0.3, 0.5, -0.6])["feasible"]
