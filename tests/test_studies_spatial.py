from aeriallab import studies_spatial as m


def test_spatial_precession_study_conserves_full_rigid_body_invariants():
    r = m.free_rotation()["summary"]
    assert r["maximum_energy_defect_j"] < 1e-8
    assert r["maximum_momentum_defect"] < 1e-8


def test_spatial_attitude_benchmark_recovers_from_declared_initial_rotation():
    r = m.attitude_recovery()
    assert r["summary"]["final_attitude_error_rad"] < 0.01
