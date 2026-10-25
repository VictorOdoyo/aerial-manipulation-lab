import numpy as np

from aeriallab import control as m


def test_hover_trim_balances_extended_arm_and_exposes_impossible_tilted_hover():
    from aeriallab.planar import acceleration, generalized_force

    q = np.array([0.0, 2.0, 0.0, 0.5, -0.6])
    r = m.hover_trim(q)
    force = generalized_force(q, r["thrust_n"], r["body_torque_nm"], r["joint_torques_nm"])
    assert r["feasible"] and np.allclose(acceleration(q, np.zeros(5), force), 0, atol=1e-12)
    assert not m.hover_trim([0, 2, 0.3, 0.5, -0.6])["feasible"]


def test_reaction_compensator_cancels_static_arm_torque_while_baseline_retains_disturbance():
    from aeriallab.planar import acceleration, generalized_force

    x = np.r_[[0.0, 2.0, 0.0, 0.5, -0.6], np.zeros(5)]
    ref = {
        "base": {"position": x[:2], "velocity": np.zeros(2), "acceleration": np.zeros(2)},
        "arm": {"position": x[3:5], "velocity": np.zeros(2), "acceleration": np.zeros(2)},
    }
    results = []
    for coupling in (True, False):
        r = m.command(x, ref, reaction_compensation=coupling)
        u = generalized_force(x[:5], r["thrust_n"], r["body_torque_nm"], r["joint_torques_nm"])
        results.append(acceleration(x[:5], x[5:], u))
    assert np.linalg.norm(results[0]) < 1e-11 and abs(results[1][2]) > 0.1


def test_position_integral_does_not_wind_up_unbounded_under_unavailable_acceleration():
    r = m.PositionIntegral()
    for _ in range(100):
        r.update([10, 10], 0.1, [3, 3], [0, 0])
    assert np.all(abs(r.value) <= 1)
