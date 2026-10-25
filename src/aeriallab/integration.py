import numpy as np


def wrench_step(
    state,
    thrust_n,
    body_torque_nm,
    joint_torques,
    dt_s,
    model=None,
    wind_velocity=(0, 0),
    contact_height=None,
):
    from .contact import plane_force
    from .core import finite, positive
    from .environment import drag
    from .planar import Model, energy, generalized_force, rhs

    p = Model() if model is None else model
    x = finite(state, real=True)
    dt = positive(dt_s)
    if x.shape != (10,):
        raise ValueError("coupled ten-state integration required")

    def derivative(extended):
        y = extended[:10]
        aero = drag(y[5:7], wind_velocity)["force_n"]
        contact = (
            np.zeros(2)
            if contact_height is None
            else plane_force(y[:5], y[5:], contact_height, model=p)["force_n"]
        )
        force = generalized_force(y[:5], thrust_n, body_torque_nm, joint_torques, aero, contact, p)
        return np.r_[rhs(y, force, p), force @ y[5:]]

    y = np.r_[x, 0.0]
    k1 = derivative(y)
    k2 = derivative(y + dt * k1 / 2)
    k3 = derivative(y + dt * k2 / 2)
    k4 = derivative(y + dt * k3)
    out = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    defect = energy(out[:10], p)["total_j"] - energy(x, p)["total_j"] - out[10]
    return {"state": out[:10], "work_j": float(out[10]), "mechanical_defect_j": float(defect)}
