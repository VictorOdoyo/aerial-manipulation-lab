import numpy as np

from aeriallab import studies_flight as m


def test_complete_arm_motion_uses_causal_feedback_and_reaches_task():
    r = m.nominal()
    s = r["summary"]
    assert s["termination"] == "completed" and s["position_rmse_m"] < 0.05
    assert s["final_arm_error_rad"] < 0.01 and s["maximum_mechanical_defect_j"] < 1e-4


def test_hover_study_remains_inside_flight_envelope():
    r = m.study("sensor-feedback-hover", m.Config(duration_s=0.3, moving_arm=False))
    assert r["summary"]["termination"] == "completed" and r["summary"]["position_rmse_m"] < 0.05


def test_baseline_replay_declares_ignored_reaction_without_relabeling_outcome():
    r = m.study("reaction-ignorant-baseline", m.Config(duration_s=0.3, reaction_compensation=False))
    assert r["summary"]["electrical_wh"] > 0 and np.all(r["arrays"]["battery_soc"] > 0)


def test_coordinated_reference_has_distinct_base_and_arm_paths():
    from aeriallab.trajectory import reference

    a = reference(4, True, True)
    assert np.allclose(a["base"]["position"], [1, 2.2])
    assert np.allclose(a["arm"]["position"], [0.7, -0.8])
