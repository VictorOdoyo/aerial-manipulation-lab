import numpy as np
import pytest

from aeriallab import quadrotor as m


def test_quadrotor_model_rejects_nonphysical_principal_inertia():
    with pytest.raises(ValueError):
        m.Model(inertia=(0.01, 0.01, 1.0))


def test_quad_allocation_has_four_independent_wrench_axes():
    a = m.allocation_matrix()
    assert np.linalg.matrix_rank(a) == 4
    assert np.allclose(a @ np.full(4, 5), [20, 0, 0, 0])


def test_full_quad_allocation_closes_hover_and_retains_failed_rotor_residual():
    r = m.allocate([20, 0, 0, 0])
    assert np.allclose(r["thrusts_n"], 5)
    failed = m.allocate([30, 0, 0, 0], maximum_n=[0, 10, 10, 10])
    assert failed["thrusts_n"][0] == 0 and np.linalg.norm(failed["residual"]) > 0.01
