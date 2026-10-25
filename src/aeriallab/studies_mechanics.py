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
