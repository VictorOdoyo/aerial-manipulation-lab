from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Model:
    mass_kg: float = 1.8
    inertia: tuple = (0.03, 0.035, 0.05)
    arm_m: float = 0.2
    yaw_ratio_m: float = 0.015
    gravity: float = 9.80665

    def __post_init__(self):
        from .core import finite, positive

        positive(self.mass_kg)
        positive(self.arm_m)
        positive(self.yaw_ratio_m)
        positive(self.gravity, allow_zero=True)
        i = finite(self.inertia, real=True)
        if i.shape != (3,) or np.any(i <= 0) or 2 * max(i) >= sum(i):
            raise ValueError("physical rigid-body principal inertias required")


def allocation_matrix(model=None):
    p = Model() if model is None else model
    arm = p.arm_m
    k = p.yaw_ratio_m
    return np.array([[1, 1, 1, 1], [0, arm, 0, -arm], [-arm, 0, arm, 0], [k, -k, k, -k]])


def allocate(wrench, maximum_n=(10, 10, 10, 10), model=None, weights=(1, 4, 4, 1)):
    from scipy.optimize import lsq_linear

    from .core import finite

    target = finite(wrench, real=True)
    maximum = finite(maximum_n, real=True)
    w = finite(weights, real=True)
    if (
        target.shape != (4,)
        or maximum.shape != (4,)
        or w.shape != (4,)
        or np.any(maximum < 0)
        or np.any(w <= 0)
    ):
        raise ValueError("four-dimensional wrench limits and weights required")
    active = maximum > 0
    forces = np.zeros(4)
    matrix = allocation_matrix(model)
    if any(active):
        forces[active] = lsq_linear(
            w[:, None] * matrix[:, active],
            w * target,
            bounds=(np.zeros(sum(active)), maximum[active]),
            tol=1e-12,
        ).x
    return {"thrusts_n": forces, "achieved": matrix @ forces, "residual": matrix @ forces - target}
