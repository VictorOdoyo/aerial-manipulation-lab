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


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def command(state, reference, model=None, reaction_compensation=True, integral_accel=(0, 0)):
    from .core import finite
    from .planar import Model, coriolis_matrix, gravity, mass_matrix

    p = Model() if model is None else model
    x = finite(state, real=True)
    integral = finite(integral_accel, real=True)
    if x.shape != (10,) or integral.shape != (2,) or type(reaction_compensation) is not bool:
        raise ValueError("estimated ten-state feedback and boolean coupling option required")
    q, v = x[:5], x[5:]
    base = reference["base"]
    arm = reference["arm"]
    base_acc = np.clip(
        np.asarray(base["acceleration"])
        + 4 * (np.asarray(base["position"]) - q[:2])
        + 3 * (np.asarray(base["velocity"]) - v[:2])
        + integral,
        -3,
        3,
    )
    arm_acc = np.clip(
        np.asarray(arm["acceleration"])
        + 60 * wrap(np.asarray(arm["position"]) - q[3:])
        + 12 * (np.asarray(arm["velocity"]) - v[3:]),
        -40,
        40,
    )
    inertia = mass_matrix(q, p)
    c = coriolis_matrix(q, v, p) @ v
    g = gravity(q, p)
    desired = np.r_[base_acc, 0.0, arm_acc]
    if reaction_compensation:
        outer = (inertia @ desired + c + g)[:2]
    else:
        outer = p.total_mass * (base_acc + np.array([0, p.gravity_m_s2]))
    pitch = float(np.clip(np.arctan2(-outer[0], outer[1]), -0.6, 0.6))
    desired[2] = 64 * float(wrap(pitch - q[2])) - 16 * v[2]
    if reaction_compensation:
        force = inertia @ desired + c + g
    else:
        force = np.r_[
            outer,
            p.base_inertia_kg_m2 * desired[2],
            inertia[3:, 3:] @ arm_acc
            + (coriolis_matrix(q, np.r_[np.zeros(3), v[3:]], p) @ np.r_[np.zeros(3), v[3:]])[3:]
            + g[3:],
        ]
    thrust = max(0, float(force[:2] @ np.array([-np.sin(q[2]), np.cos(q[2])])))
    return {
        "thrust_n": thrust,
        "body_torque_nm": float(force[2]),
        "joint_torques_nm": force[3:],
        "pitch_reference_rad": pitch,
        "base_acceleration_m_s2": base_acc,
        "generalized_reference": force,
    }


class PositionIntegral:
    def __init__(self, gain=0.3, tracking_s=0.5, maximum_m_s2=1):
        from .core import positive

        self.gain = positive(gain, allow_zero=True)
        self.tracking = positive(tracking_s)
        self.maximum = positive(maximum_m_s2)
        self.value = np.zeros(2)

    def update(self, position_error, dt_s, requested_acceleration=(0, 0), achieved_acceleration=(0, 0)):
        from .core import finite, positive

        error = finite(position_error, real=True)
        requested = finite(requested_acceleration, real=True)
        achieved = finite(achieved_acceleration, real=True)
        if error.shape != (2,) or requested.shape != (2,) or achieved.shape != (2,):
            raise ValueError("two-axis integral inputs required")
        self.value = np.clip(
            self.value + positive(dt_s) * (self.gain * error + (achieved - requested) / self.tracking),
            -self.maximum,
            self.maximum,
        )
        return self.value.copy()
