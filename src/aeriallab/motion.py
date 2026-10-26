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


def synchronized(time_s, requested_duration_s, start, stop, velocity_limit, acceleration_limit, jerk_limit):
    requested = positive(requested_duration_s)
    minimum = minimum_duration(start, stop, velocity_limit, acceleration_limit, jerk_limit)
    duration = max(requested, minimum)
    return {
        "duration_s": duration,
        "extended": duration > requested,
        **quintic(time_s, duration, start, stop),
    }


def segment_clearance(start, stop, centers, radii):
    a = finite(start, real=True)
    b = finite(stop, real=True)
    c = finite(centers, real=True, ndim=2)
    r = finite(radii, real=True, ndim=1)
    if a.shape != (2,) or b.shape != (2,) or c.shape != (len(r), 2) or len(r) == 0 or np.any(r <= 0):
        raise ValueError("planar segment and positive obstacle radii required")
    delta = b - a
    squared = delta @ delta
    fraction = np.clip(((c - a) @ delta) / squared, 0, 1) if squared > 0 else np.zeros(len(r))
    closest = a + fraction[:, None] * delta
    clearance = np.linalg.norm(c - closest, axis=1) - r
    return {"clearance_m": clearance, "collision": bool(np.any(clearance <= 0)), "closest_points": closest}


def waypoints(time_s, points, durations_s):
    p = finite(points, real=True, ndim=2)
    duration = finite(durations_s, real=True, ndim=1)
    times = np.atleast_1d(finite(time_s, real=True))
    if len(p) != len(duration) + 1 or len(duration) == 0 or np.any(duration <= 0) or times.ndim != 1:
        raise ValueError("waypoint intervals and positive durations required")
    boundaries = np.r_[0, np.cumsum(duration)]
    out = {key: [] for key in ("position", "velocity", "acceleration", "jerk")}
    for time in times:
        index = int(np.clip(np.searchsorted(boundaries, time, side="right") - 1, 0, len(duration) - 1))
        sample = quintic(time - boundaries[index], duration[index], p[index], p[index + 1])
        for key in out:
            out[key].append(sample[key])
    return {key: np.array(value) for key, value in out.items()}


def jerk_cost(start, stop, duration_s):
    a = finite(start, real=True)
    b = finite(stop, real=True)
    if a.shape != b.shape:
        raise ValueError("matching move endpoints required")
    return float(720 * np.sum((b - a) ** 2) / positive(duration_s) ** 5)
