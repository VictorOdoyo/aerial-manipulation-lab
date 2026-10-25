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
