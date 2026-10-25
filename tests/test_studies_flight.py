import numpy as np
import pytest

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


def test_gust_environment_can_deliver_nonzero_disturbance_without_changing_sensor_truth():
    from aeriallab.environment import drag, gust

    wind = gust(3, peak_m_s=6)
    force = drag([0, 0], [wind, 0])["force_n"]
    assert force[0] > 1 and force[1] == 0


def test_sensor_outage_retains_time_and_covariance_without_position_measurements():
    r = m.study("position-sensor-outage", m.Config(duration_s=0.3, position_outage=(0.1, 0.25)))
    a = r["arrays"]
    assert a["position_available"].sum() == 1 and a["time_s"][-1] == pytest.approx(0.3)
    assert np.min(np.linalg.eigvalsh(a["navigation_covariance"])) > 0


def test_outlier_study_retains_available_but_rejected_observation():
    r = m.study(
        "position-outlier-gating", m.Config(duration_s=0.3, position_outlier=True, outlier_time_s=0.1)
    )
    a = r["arrays"]
    assert np.any(a["position_available"] & ~a["position_accepted"])


def test_unmodeled_payload_produces_initial_weight_support_error():
    r = m.study("payload-model-mismatch", m.Config(duration_s=0.3, payload_kg=0.45))
    a = r["arrays"]
    assert np.min(a["state"][:, 1]) < 1.995


def test_known_payload_starts_at_consistent_hover_support():
    r = m.study("known-heavy-payload", m.Config(duration_s=0.3, payload_kg=0.45, assumed_payload_kg=0.45))
    assert r["summary"]["position_rmse_m"] < 0.05


def test_partial_loss_produces_real_actuator_saturation_residual():
    r = m.study("partial-actuator-loss", m.Config(duration_s=0.4, actuator_fault=True, fault_time_s=0.1))
    assert r["summary"]["maximum_allocation_residual"] > 0.1
