from aeriallab import studies_taskspace as m


def test_inertia():
    s = m.inertia()["summary"]
    assert s["loaded_effective_mass_kg"] > s["minimum_effective_mass_kg"]
