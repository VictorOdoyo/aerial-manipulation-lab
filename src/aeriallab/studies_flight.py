import numpy as np

from .results import curve, make
from .simulation import Config, run


def study(name, config):
    record = run(config)
    a = record["arrays"]
    t = a["time_s"]
    error = a["state"][:, :2] - a["reference_position_m"]
    arm_error = a["state"][:, 3:5] - a["reference_arm_rad"]
    for key, data in (
        ("x_m", a["state"][:, 0]),
        ("z_m", a["state"][:, 1]),
        ("reference_x_m", a["reference_position_m"][:, 0]),
        ("reference_z_m", a["reference_position_m"][:, 1]),
        ("joint1_rad", a["state"][:, 3]),
        ("joint2_rad", a["state"][:, 4]),
        ("reference_joint1_rad", a["reference_arm_rad"][:, 0]),
        ("reference_joint2_rad", a["reference_arm_rad"][:, 1]),
        ("pitch_rad", a["state"][:, 2]),
        ("position_error_m", np.linalg.norm(error, axis=1)),
        (
            "estimated_position_error_m",
            np.linalg.norm(a["estimated_state"][:, :2] - a["state"][:, :2], axis=1),
        ),
    ):
        a[key] = data
    a["valid_sample"] = t > 0
    summary = {
        "termination": record["termination"],
        "achieved_duration_s": float(t[-1]),
        "completion_fraction": float(t[-1] / config.duration_s),
        "position_rmse_m": float(np.sqrt(np.mean(error**2))),
        "arm_rmse_rad": float(np.sqrt(np.mean(arm_error**2))),
        "final_arm_error_rad": float(np.linalg.norm(arm_error[-1])),
        "electrical_wh": float(np.trapezoid(a["electric_w"], t) / 3600),
        "maximum_mechanical_defect_j": float(max(abs(a["mechanical_defect_j"]))),
        "position_rejections": int(sum(a["position_available"] & ~a["position_accepted"])),
        "maximum_allocation_residual": float(max(a["allocation_residual"])),
    }
    plots = [
        curve(
            "Achieved flight position",
            "time_s",
            [
                ("reference_x_m", "Requested x"),
                ("x_m", "Aircraft x"),
                ("reference_z_m", "Requested z"),
                ("z_m", "Aircraft z"),
            ],
            "Time (s)",
            "Position (m)",
        ),
        curve(
            "Arm motion and achieved joints",
            "time_s",
            [
                ("reference_joint1_rad", "Requested q1"),
                ("joint1_rad", "q1"),
                ("reference_joint2_rad", "Requested q2"),
                ("joint2_rad", "q2"),
            ],
            "Time (s)",
            "Joint angle (rad)",
        ),
        curve(
            "Aircraft attitude",
            "time_s",
            [("pitch_reference_rad", "Requested pitch"), ("pitch_rad", "Pitch")],
            "Time (s)",
            "Pitch (rad)",
        ),
        curve(
            "Tracking and sensor-derived position estimate",
            "time_s",
            [
                ("position_error_m", "Flight tracking norm"),
                ("estimated_position_error_m", "Observer position error"),
            ],
            "Time (s)",
            "Error (m)",
        ),
    ]
    return make(
        name,
        {
            "configuration": config.to_dict(),
            "plant": record["plant"],
            "controller_model": record["controller_model"],
            "actuators": {
                "aggregate_pairs": 2,
                "ct_n_s2": 2e-5,
                "lag_s": 0.035,
                "rated_rad_s": 1000,
                "arm_m": 0.25,
                "joint_max_nm": 2,
                "joint_slew_nm_s": 40,
            },
            "sensors": {
                "gyro_bias_rad_s": 0.005,
                "accel_bias_body_m_s2": [0.03, -0.02],
                "gyro_std_rad_s": 0.003,
                "accel_std_m_s2": 0.05,
                "position_std_m": 0.02,
                "attitude_std_rad": 0.003,
                "rotor_std_rad_s": 1,
                "position_period_s": 0.1,
                "attitude_period_s": 0.02,
                "encoder_counts": 16384,
            },
            "pack": {"series": 6, "capacity_ah": 5, "resistance_ohm": 0.06},
        },
        summary,
        a,
        plots,
        assumptions=[
            "Coupled planar five-DoF aircraft and two-link arm; spatial quaternion flight is a separate bench.",
            "Main feedback uses noisy sensor-derived state unless configuration explicitly says ideal.",
            "Motor groups represent aggregate lateral rotor pairs; quadratic thrust with exact speed lag.",
            "Illustrative physical, sensor and electrical parameters; no hardware flight qualification.",
            "Envelope violations terminate and truncate the actual trajectory; missing time is not filled as successful flight.",
        ],
    )


def nominal(scale=1, seed=42):
    return study("reaction-aware-arm-motion", Config(dt_s=0.005 / scale, seed=seed))


def hover(scale=1, seed=42):
    return study("sensor-feedback-hover", Config(dt_s=0.005 / scale, seed=seed, moving_arm=False))


def baseline(scale=1, seed=42):
    return study(
        "reaction-ignorant-baseline", Config(dt_s=0.005 / scale, seed=seed, reaction_compensation=False)
    )


def coordinated(scale=1, seed=42):
    return study("coordinated-translation", Config(dt_s=0.005 / scale, seed=seed, moving_base=True))
