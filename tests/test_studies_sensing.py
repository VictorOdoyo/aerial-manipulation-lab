import numpy as np

from aeriallab import studies_sensing as m


def test_inertial_outage_study_retains_absent_position_measurements_and_psd_covariance():
    r = m.outage()
    a = r["arrays"]
    selected = (a["time_s"] >= 2) & (a["time_s"] <= 4)
    assert not np.any(a["position_available"][selected])
    assert np.min(np.linalg.eigvalsh(a["covariance"])) > 0
