from aeriallab import studies_flight as m


def test_complete_arm_motion_uses_causal_feedback_and_reaches_task():
    r = m.nominal()
    s = r["summary"]
    assert s["termination"] == "completed" and s["position_rmse_m"] < 0.05
    assert s["final_arm_error_rad"] < 0.01 and s["maximum_mechanical_defect_j"] < 1e-4
