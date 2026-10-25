from aeriallab import studies_spatial as m


def test_spatial_precession_study_conserves_full_rigid_body_invariants():
    r = m.free_rotation()["summary"]
    assert r["maximum_energy_defect_j"] < 1e-8
    assert r["maximum_momentum_defect"] < 1e-8
