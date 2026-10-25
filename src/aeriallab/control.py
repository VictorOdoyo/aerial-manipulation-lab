import numpy as np


def hover_trim(q, model=None, wind_n=(0, 0), tip_force_n=(0, 0)):
    from .planar import Model, configuration, generalized_force, gravity

    p = Model() if model is None else model
    x = configuration(q)
    target = gravity(x, p) - generalized_force(x, 0, wind_n=wind_n, tip_force_n=tip_force_n, model=p)
    matrix = np.zeros((5, 4))
    matrix[:2, 0] = [-np.sin(x[2]), np.cos(x[2])]
    matrix[2:, 1:] = np.eye(3)
    u = np.linalg.lstsq(matrix, target, rcond=None)[0]
    residual = matrix @ u - target
    return {
        "thrust_n": max(0, float(u[0])),
        "body_torque_nm": float(u[1]),
        "joint_torques_nm": u[2:],
        "residual": residual,
        "feasible": bool(u[0] >= 0 and np.linalg.norm(residual) < 1e-8),
    }
