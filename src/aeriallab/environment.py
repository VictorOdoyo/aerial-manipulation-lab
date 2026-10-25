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
