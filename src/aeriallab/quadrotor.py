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
