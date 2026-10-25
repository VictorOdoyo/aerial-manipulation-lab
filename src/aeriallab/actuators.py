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
    maximum = finite(maximum_rad_s, real=True)
    if np.any(maximum < 0):
        raise ValueError("nonnegative rotor speed ceilings required")
    if speed.shape != command.shape or np.any(speed < 0) or np.any(command < 0):
        raise ValueError("aligned bounded positive rotor states and commands required")
    target = np.minimum(np.sqrt(command / ct), maximum)
    next_speed = target + (speed - target) * np.exp(-dt / tau)
    return {"speed_rad_s": next_speed, "thrust_n": ct * next_speed**2, "limited": command > ct * maximum**2}


def speed_ceiling(bus_v, nominal_v=22.2, rated_rad_s=1000, health=(1, 1)):
    from .core import finite, positive

    health = finite(health, real=True)
    if health.shape != (2,) or np.any((health < 0) | (health > 1)):
        raise ValueError("two bounded rotor-pair health fractions required")
    fraction = min(1.0, positive(bus_v, allow_zero=True) / positive(nominal_v))
    return positive(rated_rad_s) * fraction * np.sqrt(health)


def joint_effort(request, previous, dt_s, maximum_nm=2, slew_nm_s=40):
    from .core import finite, positive

    r = finite(request, real=True)
    old = finite(previous, real=True)
    dt = positive(dt_s)
    limit = positive(maximum_nm)
    rate = positive(slew_nm_s)
    if r.shape != (2,) or old.shape != (2,):
        raise ValueError("two joint effort commands required")
    command = np.clip(r, -limit, limit)
    applied = old + np.clip(command - old, -rate * dt, rate * dt)
    return {"torque_nm": applied, "limited": bool(np.any(abs(applied - r) > 1e-8))}


def mean_thrust(
    speed_rad_s, thrust_command_n, dt_s, coefficient=2e-5, time_constant_s=0.035, maximum_rad_s=1000
):
    from .core import finite, positive

    old = finite(speed_rad_s, real=True)
    command = finite(thrust_command_n, real=True)
    dt = positive(dt_s)
    ct = positive(coefficient)
    tau = positive(time_constant_s)
    maximum = finite(maximum_rad_s, real=True)
    if old.shape != command.shape or np.any(old < 0) or np.any(command < 0) or np.any(maximum < 0):
        raise ValueError("aligned nonnegative rotor states commands and ceilings required")
    target = np.minimum(np.sqrt(command / ct), maximum)
    delta = old - target
    decay = np.exp(-dt / tau)
    return ct * (
        target**2 + 2 * target * delta * tau * (1 - decay) / dt + delta**2 * tau * (1 - decay**2) / (2 * dt)
    )
