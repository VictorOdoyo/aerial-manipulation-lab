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
