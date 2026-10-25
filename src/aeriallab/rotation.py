import numpy as np


def normalize(quaternion):
    from .core import finite

    q = finite(quaternion, real=True)
    if q.shape != (4,) or np.linalg.norm(q) < 1e-12:
        raise ValueError("nonzero scalar-first quaternion required")
    return q / np.linalg.norm(q)


def multiply(left, right):
    a = normalize(left)
    b = normalize(right)
    w = a[0] * b[0] - a[1:] @ b[1:]
    return normalize(np.r_[w, a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])])


def matrix(quaternion):
    w, x, y, z = normalize(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def integrate(quaternion, omega_rad_s, dt_s):
    from .core import finite, positive

    q = normalize(quaternion)
    omega = finite(omega_rad_s, real=True)
    dt = positive(dt_s)
    if omega.shape != (3,):
        raise ValueError("three body angular rates required")
    angle = np.linalg.norm(omega) * dt
    vector = omega * dt / 2 * np.sinc(angle / (2 * np.pi))
    increment = np.r_[np.cos(angle / 2), vector]
    return multiply(q, increment)


def error(current, desired):
    q = normalize(current)
    target = normalize(desired)
    relative = multiply(q * np.array([1, -1, -1, -1]), target)
    if relative[0] < 0:
        relative = -relative
    norm = np.linalg.norm(relative[1:])
    if norm < 1e-12:
        return 2 * relative[1:]
    return relative[1:] / norm * (2 * np.arctan2(norm, relative[0]))
