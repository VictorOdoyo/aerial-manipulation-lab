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
    from .actuators import allocate, wrench_error_n
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
            residual.append(wrench_error_n(allocation["residual"]))
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
                "Force-equivalent wrench residual (N)",
            ),
        ],
        assumptions=[
            "Torque residual is divided by the 0.25 m lever arm before combining with force; allocator weights are [1,4]."
        ],
    )


def rotor_lag(scale=1, seed=42):
    from .actuators import mean_thrust, rotor_step, wrench

    dt = 0.001 / scale
    t = np.arange(round(0.4 / dt) + 1) * dt
    omega = np.zeros(2)
    actual = []
    requested = []
    for time in t:
        target = (
            np.array([0.0, 0.0])
            if time < 0.05
            else np.array([12.0, 12.0])
            if time < 0.2
            else np.array([10.0, 14.0])
        )
        average = mean_thrust(omega, target, dt)
        omega = rotor_step(omega, target, dt)["speed_rad_s"]
        actual.append(wrench(average))
        requested.append(wrench(target))
    actual = np.array(actual)
    requested = np.array(requested)
    a = {
        "time_s": t,
        "actual_force_n": actual[:, 0],
        "requested_force_n": requested[:, 0],
        "actual_torque_nm": actual[:, 1],
        "requested_torque_nm": requested[:, 1],
    }
    return make(
        "rotor-lag-response",
        {"lag_s": 0.035, "ct_n_s2": 2e-5, "dt_s": dt},
        {"final_force_n": float(actual[-1, 0]), "final_torque_nm": float(actual[-1, 1])},
        a,
        [
            curve(
                "Total thrust interval average",
                "time_s",
                [("requested_force_n", "Requested"), ("actual_force_n", "Delivered")],
                "Time (s)",
                "Force (N)",
            ),
            curve(
                "Differential pitch torque",
                "time_s",
                [("requested_torque_nm", "Requested"), ("actual_torque_nm", "Delivered")],
                "Time (s)",
                "Torque (Nm)",
            ),
        ],
    )


def payload_sequence(scale=1, seed=42):
    from .payload import pickup, release
    from .planar import Model, energy, linear_momentum

    p = Model(payload_kg=0, gravity_m_s2=0)
    initial = np.r_[[0.0, 2.0, 0.2, 0.4, -0.3], [1.0, -0.2, 0.3, 0.4, -0.2]]
    masses = np.linspace(0.01, 0.5, 61 * scale)
    loss = []
    momentum = []
    energy_defect = []
    vx = []
    initial_energy = energy(initial, p)["total_j"]
    initial_momentum = linear_momentum(initial[:5], initial[5:], p)
    for mass in masses:
        captured = pickup(initial, float(mass), model=p)
        released = release(captured["state"], float(mass), captured["model"])
        x = released["state"]
        final_momentum = (
            linear_momentum(x[:5], x[5:], released["model"]) + mass * released["object_velocity_m_s"]
        )
        final_energy = energy(x, released["model"])["total_j"] + 0.5 * mass * np.sum(
            released["object_velocity_m_s"] ** 2
        )
        loss.append(captured["dissipated_j"])
        momentum.append(np.linalg.norm(final_momentum - initial_momentum))
        energy_defect.append(initial_energy - final_energy - captured["dissipated_j"])
        vx.append(x[5])
    a = {
        "payload_kg": masses,
        "capture_loss_j": loss,
        "momentum_defect_n_s": momentum,
        "energy_defect_j": energy_defect,
        "base_velocity_x_m_s": vx,
    }
    return make(
        "payload-capture-release",
        {"model": asdict(p), "initial_state": initial.tolist(), "object_velocity_m_s": [0, 0]},
        {
            "maximum_momentum_defect_n_s": float(max(momentum)),
            "maximum_energy_defect_j": float(max(abs(np.array(energy_defect)))),
        },
        a,
        [
            curve(
                "Capture dissipates relative kinetic energy",
                "payload_kg",
                [("capture_loss_j", "Inelastic loss")],
                "Added and released payload (kg)",
                "Energy (J)",
            ),
            curve(
                "Aircraft velocity after pickup and release",
                "payload_kg",
                [("base_velocity_x_m_s", "Base x velocity")],
                "Payload (kg)",
                "Velocity (m/s)",
            ),
        ],
    )


def contact_restitution(scale=1, seed=42):
    from .contact import normal_impulse
    from .planar import tip_velocity

    q = np.array([0.0, 1.0, 0.2, 0.3, -0.4])
    v = np.array([1.0, -2.0, 0.3, 0.4, -0.2])
    incoming = tip_velocity(q, v)[1]
    e = np.linspace(0, 1, 101 * scale)
    speed = []
    loss = []
    impulse = []
    for value in e:
        r = normal_impulse(q, v, float(value))
        speed.append(tip_velocity(q, r["velocity"])[1])
        loss.append(r["dissipated_j"])
        impulse.append(r["impulse_n_s"])
    a = {
        "restitution": e,
        "outgoing_m_s": speed,
        "analytic_outgoing_m_s": -e * incoming,
        "dissipated_j": loss,
        "impulse_n_s": impulse,
    }
    return make(
        "contact-impact-restitution",
        {"configuration": q.tolist(), "incoming_velocity": v.tolist(), "model": "default planar"},
        {
            "maximum_velocity_residual_m_s": float(max(abs(np.array(speed) + e * incoming))),
            "elastic_energy_loss_j": float(loss[-1]),
        },
        a,
        [
            curve(
                "Normal impact law",
                "restitution",
                [("analytic_outgoing_m_s", "Analytic -e v_in"), ("outgoing_m_s", "Coupled impulse")],
                "Restitution",
                "Outgoing normal speed (m/s)",
            ),
            curve(
                "Impact energy accounting",
                "restitution",
                [("dissipated_j", "Dissipated")],
                "Restitution",
                "Energy (J)",
            ),
        ],
    )


def manipulability_curve(scale=1, seed=42):
    from .arm import manipulability
    from .planar import Model

    p = Model()
    angle = np.linspace(0, np.pi, 201 * scale)
    data = [manipulability([0, 2, 0, 0.3, q2], p) for q2 in angle]
    area = np.array([r["area_m2"] for r in data])
    singular = np.array([r["singular_values_m"] for r in data])
    exact = p.link1_m * p.link2_m * abs(np.sin(angle))
    a = {
        "joint2_rad": angle,
        "area_m2": area,
        "analytic_area_m2": exact,
        "largest_m": singular[:, 0],
        "smallest_m": singular[:, 1],
    }
    return make(
        "arm-manipulability",
        {"model": asdict(p), "joint1_rad": 0.3},
        {"maximum_area_residual_m2": float(max(abs(area - exact)))},
        a,
        [
            curve(
                "Task-space area metric",
                "joint2_rad",
                [("analytic_area_m2", "Analytic l1 l2 |sin q2|"), ("area_m2", "Jacobian determinant")],
                "Relative q2 (rad)",
                "Area metric (m squared)",
            ),
            curve(
                "Task observability",
                "joint2_rad",
                [("largest_m", "Largest singular value"), ("smallest_m", "Smallest singular value")],
                "Relative q2 (rad)",
                "Jacobian singular value (m)",
            ),
        ],
    )
