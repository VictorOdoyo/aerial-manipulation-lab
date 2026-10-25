import pytest

from aeriallab import planar as m


def test_planar_model_has_positive_body_and_link_properties():
    p = m.Model()
    assert p.total_mass == pytest.approx(2.33)
    with pytest.raises(ValueError):
        m.Model(link1_m=-1)
