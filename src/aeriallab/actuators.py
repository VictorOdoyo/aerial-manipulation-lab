import numpy as np


def wrench(thrusts_n, arm_m=0.25):
    from .core import finite, positive

    f = finite(thrusts_n, real=True)
    if f.shape != (2,) or np.any(f < 0):
        raise ValueError("two nonnegative aggregate rotor-pair thrusts required")
    arm = positive(arm_m)
    return np.array([sum(f), arm * (f[1] - f[0])])
