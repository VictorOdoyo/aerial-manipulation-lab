from aeriallab import studies_mechanics as m


def test_energy_convergence_reduces_global_conservation_error():
    r = m.energy_convergence()
    assert r["summary"]["fine_error_j"] < r["summary"]["coarse_error_j"] / 100
