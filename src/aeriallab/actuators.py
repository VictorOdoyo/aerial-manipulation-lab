import numpy as np


def wrench(thrusts_n, arm_m=0.25):
    from .core import finite, positive

    f = finite(thrusts_n, real=True)
    if f.shape != (2,) or np.any(f < 0):
        raise ValueError("two nonnegative aggregate rotor-pair thrusts required")
    arm = positive(arm_m)
    return np.array([sum(f), arm * (f[1] - f[0])])


def allocate(thrust_n, torque_nm, arm_m=0.25, maximum_n=(20, 20), weights=(1, 4)):
    from scipy.optimize import lsq_linear

    from .core import finite, positive

    arm = positive(arm_m)
    maximum = finite(maximum_n, real=True)
    w = finite(weights, real=True)
    target = finite([thrust_n, torque_nm], real=True)
    if maximum.shape != (2,) or w.shape != (2,) or np.any(maximum < 0) or np.any(w <= 0):
        raise ValueError("two nonnegative thrust capacities and positive wrench weights required")
    matrix = np.array([[1, 1], [-arm, arm]])
    active = maximum > 0
    forces = np.zeros(2)
    if np.any(active):
        forces[active] = lsq_linear(
            matrix[:, active] * w[:, None],
            target * w,
            bounds=(np.zeros(sum(active)), maximum[active]),
            tol=1e-12,
        ).x
    achieved = matrix @ forces
    return {
        "thrusts_n": forces,
        "achieved": achieved,
        "residual": achieved - target,
        "limited": bool(np.linalg.norm(achieved - target) > 1e-6),
    }


def rotor_step(
    speed_rad_s, thrust_command_n, dt_s, coefficient=2e-5, time_constant_s=0.035, maximum_rad_s=1000
):
    from .core import finite, positive

    speed = finite(speed_rad_s, real=True)
    command = finite(thrust_command_n, real=True)
    dt = positive(dt_s)
    ct = positive(coefficient)
    tau = positive(time_constant_s)
    maximum = positive(maximum_rad_s)
    if (
        speed.shape != command.shape
        or np.any(speed < 0)
        or np.any(command < 0)
        or np.any(speed > maximum * (1 + 1e-8))
    ):
        raise ValueError("aligned bounded positive rotor states and commands required")
    target = np.minimum(np.sqrt(command / ct), maximum)
    next_speed = target + (speed - target) * np.exp(-dt / tau)
    return {"speed_rad_s": next_speed, "thrust_n": ct * next_speed**2, "limited": command > ct * maximum**2}
