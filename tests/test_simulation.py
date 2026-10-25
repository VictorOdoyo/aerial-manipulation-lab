import pytest

from aeriallab import simulation as m


def test_replay_config_rejects_ambiguous_switches_and_unresolved_timing():
    with pytest.raises(ValueError):
        m.Config(dt_s=0.1)
    with pytest.raises(ValueError):
        m.Config(moving_arm=1)
    with pytest.raises(ValueError):
        m.Config(position_outage=(3, 2))
    assert m.Config().feedback == "estimated"
