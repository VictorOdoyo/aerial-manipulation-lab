import numpy as np


def inverse_kinematics(base_pose, target_world_m, elbow=1, model=None):
    from .core import finite
    from .planar import Model

    p = Model() if model is None else model
    base = finite(base_pose, real=True)
    target = finite(target_world_m, real=True)
    if base.shape != (3,) or target.shape != (2,) or elbow not in (-1, 1):
        raise ValueError("planar base pose target and signed elbow branch required")
    dx, dz = target - base[:2]
    theta = base[2]
    local = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) @ np.array([dx, dz])
    u, v = local[0], -local[1]
    cosine = (u * u + v * v - p.link1_m**2 - p.link2_m**2) / (2 * p.link1_m * p.link2_m)
    if abs(cosine) > 1 + 1e-10:
        raise ValueError("end-effector target outside arm workspace")
    q2 = elbow * np.arccos(np.clip(cosine, -1, 1))
    q1 = np.arctan2(u, v) - np.arctan2(p.link2_m * np.sin(q2), p.link1_m + p.link2_m * np.cos(q2))
    return np.array([q1, q2])


def manipulability(q, model=None):
    from .planar import kinematics

    jac = kinematics(q, model)["tip"][1][:, 3:]
    singular = np.linalg.svd(jac, compute_uv=False)
    return {
        "area_m2": float(abs(np.linalg.det(jac))),
        "singular_values_m": singular,
        "singular": bool(singular[-1] < 1e-8),
    }
