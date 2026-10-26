import numpy as np

from aeriallab import studies_taskspace as m


def test_inertia():
    s = m.inertia()["summary"]
    assert s["loaded_effective_mass_kg"] > s["minimum_effective_mass_kg"]


def test_velocity():
    r = m.velocity()
    assert np.all(r["arrays"]["dynamic_j"] <= r["arrays"]["euclidean_j"] + 1e-12)
