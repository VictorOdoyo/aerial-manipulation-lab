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


def normal_impulse(q, velocity, restitution=0, model=None):
    from .planar import configuration, kinematics, mass_matrix

    x = configuration(q)
    v = configuration(velocity)
    if not 0 <= restitution <= 1:
        raise ValueError("restitution in [0,1] required")
    jac = kinematics(x, model)["tip"][1][1]
    normal = float(jac @ v)
    if normal >= 0:
        return {"velocity": v.copy(), "impulse_n_s": 0.0, "dissipated_j": 0.0}
    matrix = mass_matrix(x, model)
    inverse = np.linalg.solve(matrix, jac)
    impulse = -(1 + restitution) * normal / (jac @ inverse)
    out = v + inverse * impulse
    return {
        "velocity": out,
        "impulse_n_s": float(impulse),
        "dissipated_j": float(0.5 * v @ matrix @ v - 0.5 * out @ matrix @ out),
    }


def joint_stop(q, velocity, joint=0, restitution=0, model=None):
    from .core import integer
    from .planar import configuration, mass_matrix

    x = configuration(q)
    v = configuration(velocity)
    integer(joint, 0)
    if joint > 1 or not 0 <= restitution <= 1:
        raise ValueError("valid arm joint and restitution required")
    matrix = mass_matrix(x, model)
    jac = np.eye(5)[joint + 3]
    inverse = np.linalg.solve(matrix, jac)
    impulse = -(1 + restitution) * (jac @ v) / (jac @ inverse)
    out = v + inverse * impulse
    return {
        "velocity": out,
        "impulse_nm_s": float(impulse),
        "dissipated_j": float(0.5 * v @ matrix @ v - 0.5 * out @ matrix @ out),
    }
