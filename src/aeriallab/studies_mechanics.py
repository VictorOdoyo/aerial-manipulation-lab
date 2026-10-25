from dataclasses import asdict

import numpy as np

from .results import curve, make


def energy_convergence(scale=1, seed=42):
    from .planar import Model, energy, step

    p = Model()
    initial = np.r_[[0.0, 2.0, 0.2, 0.4, -0.3], [0.1, -0.2, 0.3, 1.0, -0.8]]
    steps = np.array([0.04, 0.02, 0.01, 0.005]) / scale
    errors = []
    for dt in steps:
        x = initial.copy()
        e = energy(x, p)["total_j"]
        for _ in range(round(0.4 / dt)):
            x = step(x, np.zeros(5), dt, p)
        errors.append(abs(energy(x, p)["total_j"] - e))
    a = {
        "dt_s": steps,
        "energy_error_j": errors,
        "order4_reference_j": errors[0] * (steps / steps[0]) ** 4,
        "positive_error": np.array(errors) > 0,
    }
    return make(
        "coupled-energy-convergence",
        {"model": asdict(p), "initial_state": initial.tolist(), "duration_s": 0.4},
        {"coarse_error_j": errors[0], "fine_error_j": errors[-1]},
        a,
        [
            curve(
                "Mechanical energy error under refinement",
                "dt_s",
                [("energy_error_j", "Measured"), ("order4_reference_j", "Expected order 4 slope")],
                "Time step (s)",
                "Absolute energy error (J)",
                xscale="log",
                yscale="log",
                valid="positive_error",
            )
        ],
        assumptions=[
            "Free conservative floating-base model without ground collision; reference slope is expected RK4 order, not a fitted device model."
        ],
    )


def internal_reaction(scale=1, seed=42):
    from .planar import Model, linear_momentum, step

    p = Model(gravity_m_s2=0)
    dt = 0.0025 / scale
    t = np.arange(round(0.5 / dt) + 1) * dt
    x = np.r_[[0.0, 2.0, 0.0, 0.4, -0.3], np.zeros(5)]
    history = [x.copy()]
    momentum = [linear_momentum(x[:5], x[5:], p)]
    for _ in t[1:]:
        x = step(x, [0, 0, 0, 0.05, -0.04], dt, p)
        history.append(x.copy())
        momentum.append(linear_momentum(x[:5], x[5:], p))
    history = np.array(history)
    momentum = np.array(momentum)
    a = {
        "time_s": t,
        "state": history,
        "pitch_rad": history[:, 2],
        "joint1_rad": history[:, 3],
        "joint2_rad": history[:, 4],
        "momentum_x": momentum[:, 0],
        "momentum_z": momentum[:, 1],
    }
    return make(
        "internal-reaction-momentum",
        {"model": asdict(p), "joint_torques_nm": [0.05, -0.04], "dt_s": dt},
        {
            "maximum_momentum_defect_n_s": float(max(np.linalg.norm(momentum - momentum[0], axis=1))),
            "final_pitch_rad": float(history[-1, 2]),
        },
        a,
        [
            curve(
                "Internal arm motion reacts on the floating base",
                "time_s",
                [
                    ("pitch_rad", "Aircraft pitch"),
                    ("joint1_rad", "Relative q1"),
                    ("joint2_rad", "Relative q2"),
                ],
                "Time (s)",
                "Angle (rad)",
            ),
            curve(
                "External linear momentum remains zero",
                "time_s",
                [("momentum_x", "x"), ("momentum_z", "z")],
                "Time (s)",
                "Momentum (N s)",
                ylim=(-1e-7, 1e-7),
            ),
        ],
    )


def trim_workspace(scale=1, seed=42):
    from .actuators import allocate
    from .control import hover_trim
    from .planar import Model

    angle = np.linspace(-0.9, 0.9, 101 * scale)
    a = {"joint1_rad": angle}
    summary = {}
    for label, payload in (("nominal", 0.1), ("heavy", 0.8)):
        p = Model(payload_kg=payload)
        forces = []
        residual = []
        for q1 in angle:
            r = hover_trim([0, 2, 0, q1, -q1], p)
            allocation = allocate(r["thrust_n"], r["body_torque_nm"])
            forces.append(allocation["thrusts_n"])
            residual.append(np.linalg.norm(allocation["residual"]))
        forces = np.array(forces)
        a[label + "_left_n"] = forces[:, 0]
        a[label + "_right_n"] = forces[:, 1]
        a[label + "_residual"] = np.array(residual)
        summary[label + "_feasible_fraction"] = float(np.mean(np.array(residual) < 1e-6))
    return make(
        "hover-trim-workspace",
        {"payloads_kg": [0.1, 0.8], "maximum_pair_thrust_n": [20, 20], "arm_pose_rule": "q2=-q1"},
        summary,
        a,
        [
            curve(
                "Allocated aggregate rotor pairs",
                "joint1_rad",
                [
                    ("nominal_left_n", "Nominal left"),
                    ("nominal_right_n", "Nominal right"),
                    ("heavy_left_n", "Heavy left"),
                    ("heavy_right_n", "Heavy right"),
                ],
                "Relative q1 (rad)",
                "Thrust (N)",
            ),
            curve(
                "Unattainable hover wrench remains visible",
                "joint1_rad",
                [("nominal_residual", "Nominal payload"), ("heavy_residual", "Heavy payload")],
                "Relative q1 (rad)",
                "Wrench residual norm (mixed N / Nm)",
            ),
        ],
        assumptions=[
            "Residual norm combines stated force and torque coordinates; allocator uses explicit [1,4] wrench weights."
        ],
    )
