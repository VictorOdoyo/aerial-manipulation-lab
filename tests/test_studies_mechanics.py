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
