import numpy as np


def plane_force(q, velocity, height_m=0, stiffness_n_m=500, damping_n_s_m=20, model=None):
    from .core import positive
    from .planar import kinematics, tip_velocity

    point = kinematics(q, model)["tip"][0]
    speed = tip_velocity(q, velocity, model)
    penetration = max(0, height_m - point[1])
    if not np.isfinite(height_m):
        raise ValueError("finite contact plane required")
    normal = (
        max(0, positive(stiffness_n_m) * penetration - positive(damping_n_s_m, allow_zero=True) * speed[1])
        if penetration > 0
        else 0.0
    )
    return {
        "force_n": np.array([0.0, normal]),
        "penetration_m": penetration,
        "spring_j": 0.5 * stiffness_n_m * penetration**2,
    }
