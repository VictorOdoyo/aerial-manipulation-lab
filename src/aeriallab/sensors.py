import numpy as np


def rotation(pitch_rad):
    if not np.isfinite(pitch_rad):
        raise ValueError("finite pitch required")
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    return np.array([[c, -s], [s, c]])


def specific_force(acceleration_world, pitch_rad, gravity=9.80665):
    from .core import finite, positive

    a = finite(acceleration_world, real=True)
    if a.shape != (2,):
        raise ValueError("planar world acceleration required")
    return rotation(pitch_rad).T @ (a + np.array([0, positive(gravity, allow_zero=True)]))
