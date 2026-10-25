import numpy as np


def propulsion_power(
    thrusts_n, rotor_speeds, air_density=1.225, disk_area_m2=0.08, efficiency=0.65, profile_coefficient=2e-8
):
    from .core import finite, positive

    f = finite(thrusts_n, real=True)
    omega = finite(rotor_speeds, real=True)
    if f.shape != omega.shape or np.any(f < 0) or np.any(omega < 0) or not 0 < efficiency <= 1:
        raise ValueError("aligned nonnegative rotor states and bounded efficiency required")
    induced = f**1.5 / np.sqrt(2 * positive(air_density) * positive(disk_area_m2))
    profile = positive(profile_coefficient, allow_zero=True) * omega**3
    return {
        "shaft_w": float(sum(induced + profile)),
        "electrical_w": float(sum(induced + profile) / efficiency),
    }


def servo_power(torque_nm, speed_rad_s, efficiency=0.7, idle_per_joint_w=3):
    from .core import finite, positive

    torque = finite(torque_nm, real=True)
    speed = finite(speed_rad_s, real=True)
    if torque.shape != (2,) or speed.shape != (2,) or not 0 < efficiency <= 1:
        raise ValueError("two joint states and efficiency in (0,1] required")
    mechanical = torque * speed
    draw = positive(idle_per_joint_w, allow_zero=True) + np.maximum(mechanical, 0) / efficiency
    return {
        "mechanical_w": float(sum(mechanical)),
        "electrical_w": float(sum(draw)),
        "heat_w": float(sum(draw - mechanical)),
    }
