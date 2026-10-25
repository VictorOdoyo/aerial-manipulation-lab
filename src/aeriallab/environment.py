import numpy as np


def drag(velocity, wind=(0, 0), air_density=1.225, drag_area=(0.08, 0.12)):
    from .core import finite, positive

    v = finite(velocity, real=True)
    w = finite(wind, real=True)
    area = finite(drag_area, real=True)
    if v.shape != (2,) or w.shape != (2,) or area.shape != (2,) or np.any(area < 0):
        raise ValueError("planar wind velocity and nonnegative drag areas required")
    relative = v - w
    force = -0.5 * positive(air_density) * area * abs(relative) * relative
    return {"force_n": force, "relative_dissipation_w": float(-force @ relative)}


def gust(time_s, start_s=2, duration_s=2, peak_m_s=5):
    from .core import finite, positive

    t = finite(time_s, real=True)
    duration = positive(duration_s)
    if not np.isfinite(start_s + peak_m_s):
        raise ValueError("finite gust parameters required")
    phase = (t - start_s) / duration
    active = (phase >= 0) & (phase <= 1)
    return np.where(active, peak_m_s * np.sin(np.pi * np.clip(phase, 0, 1)) ** 2, 0)
