from aeriallab import studies_mechanics as m


def test_energy_convergence_reduces_global_conservation_error():
    r = m.energy_convergence()
    assert r["summary"]["fine_error_j"] < r["summary"]["coarse_error_j"] / 100


def test_internal_reaction_study_has_base_motion_without_external_momentum():
    r = m.internal_reaction()
    assert abs(r["summary"]["final_pitch_rad"]) > 0.001
    assert r["summary"]["maximum_momentum_defect_n_s"] < 1e-7


def test_heavy_payload_reduces_attainable_hover_workspace():
    r = m.trim_workspace()["summary"]
    assert r["heavy_feasible_fraction"] < r["nominal_feasible_fraction"]


def test_rotor_lag_study_retains_transient_wrench_error():
    a = m.rotor_lag()["arrays"]
    assert max(abs(a["requested_force_n"] - a["actual_force_n"])) > 10
    assert abs(a["actual_torque_nm"][-1] - 1) < 0.02


def test_payload_sequence_conserves_momentum_and_accounts_for_capture_loss():
    r = m.payload_sequence()["summary"]
    assert r["maximum_momentum_defect_n_s"] < 1e-12
    assert r["maximum_energy_defect_j"] < 1e-12


def test_contact_restitution_study_recovers_impact_law_and_elastic_energy_limit():
    r = m.contact_restitution()["summary"]
    assert r["maximum_velocity_residual_m_s"] < 1e-12
    assert abs(r["elastic_energy_loss_j"]) < 1e-12


def test_manipulability_study_matches_independent_planar_formula():
    assert m.manipulability_curve()["summary"]["maximum_area_residual_m2"] < 1e-12


def test_internal_joint_stop_study_conserves_external_momentum():
    assert m.joint_impact()["summary"]["maximum_momentum_defect_n_s"] < 1e-12
