import numpy as np

from .core import finite, positive


def quintic(time_s, duration_s, start, stop):
    t = finite(time_s, real=True)
    duration = positive(duration_s)
    a = np.atleast_1d(finite(start, real=True))
    b = np.atleast_1d(finite(stop, real=True))
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("aligned endpoint coordinates required")
    u = np.clip(t / duration, 0, 1)
    delta = b - a
    active = (t >= 0) & (t <= duration)
    shape = 10 * u**3 - 15 * u**4 + 6 * u**5
    velocity = np.where(active, 30 * u**2 - 60 * u**3 + 30 * u**4, 0) / duration
    acceleration = np.where(active, 60 * u - 180 * u**2 + 120 * u**3, 0) / duration**2
    jerk = np.where(active, 60 - 360 * u + 360 * u**2, 0) / duration**3
    return {
        "position": a + shape[..., None] * delta,
        "velocity": velocity[..., None] * delta,
        "acceleration": acceleration[..., None] * delta,
        "jerk": jerk[..., None] * delta,
    }


def minimum_duration(start, stop, velocity_limit, acceleration_limit, jerk_limit):
    a = np.atleast_1d(finite(start, real=True))
    b = np.atleast_1d(finite(stop, real=True))
    delta = abs(b - a)
    limits = [
        np.broadcast_to(finite(v, real=True), a.shape)
        for v in (velocity_limit, acceleration_limit, jerk_limit)
    ]
    if a.shape != b.shape or a.ndim != 1 or any(np.any(v <= 0) for v in limits):
        raise ValueError("aligned endpoints and positive component limits required")
    bounds = np.stack(
        [
            1.875 * delta / limits[0],
            np.sqrt((10 * np.sqrt(3) / 3) * delta / limits[1]),
            np.cbrt(60 * delta / limits[2]),
        ]
    )
    return float(max(np.max(bounds), 1e-9))
