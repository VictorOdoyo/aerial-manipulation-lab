import numpy as np

from aeriallab import studies_sensing as m


def test_inertial_outage_study_retains_absent_position_measurements_and_psd_covariance():
    r = m.outage()
    a = r["arrays"]
    selected = (a["time_s"] >= 2) & (a["time_s"] <= 4)
    assert not np.any(a["position_available"][selected])
    assert np.min(np.linalg.eigvalsh(a["covariance"])) > 0


def test_observability_study_retains_rank_deficiency_until_third_position_sample():
    r = m.observability()["summary"]
    assert r["two_measurement_rank"] == 4 and r["three_measurement_rank"] == 6
