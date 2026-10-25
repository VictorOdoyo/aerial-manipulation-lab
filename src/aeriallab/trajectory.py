import numpy as np


def quintic(time_s, start, end, duration_s, start_time_s=0):
    from .core import finite, positive

    t = finite(time_s, real=True)
    duration = positive(duration_s)
    a = finite(start, real=True)
    b = finite(end, real=True)
    if a.shape != b.shape or not np.isfinite(start_time_s):
        raise ValueError("matching finite trajectory endpoints required")
    u = np.clip((t - start_time_s) / duration, 0, 1)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
    dds = (60 * u - 180 * u**2 + 120 * u**3) / duration**2
    return {
        "position": a + s[..., None] * (b - a),
        "velocity": ds[..., None] * (b - a),
        "acceleration": dds[..., None] * (b - a),
    }


def waypoints(time_s, knots_s, positions):
    from .core import finite, timebase

    t = float(finite(time_s, real=True))
    knots = timebase(knots_s)
    p = finite(positions, real=True, ndim=2)
    if len(p) != len(knots):
        raise ValueError("one position per waypoint clock required")
    index = int(np.clip(np.searchsorted(knots, t, side="right") - 1, 0, len(knots) - 2))
    return quintic(t, p[index], p[index + 1], knots[index + 1] - knots[index], knots[index])


def reference(time_s, moving_base=False, moving_arm=True):
    from .core import finite

    t = float(finite(time_s, real=True))
    base = quintic(t, [0, 2], [1, 2.2] if moving_base else [0, 2], 3, 1)
    arm = quintic(t, [0, 0], [0.7, -0.8] if moving_arm else [0, 0], 2, 1)
    return {"base": base, "arm": arm}
