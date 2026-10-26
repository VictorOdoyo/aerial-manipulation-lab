import numpy as np

from .results import curve, make


def inertia(scale=1, seed=42):
    from .planar import Model, mass_matrix
    from .taskspace import operational_inertia, tip

    payload = np.linspace(0, 1, 101 * scale)
    values = []
    q = np.array([0, 1, 0, 0.4, -0.7])
    for mass in payload:
        p = Model(payload_kg=float(mass))
        values.append(np.linalg.eigvalsh(operational_inertia(mass_matrix(q, p), tip(q, p)[1])))
    values = np.array(values)
    return make(
        "task-inertia-payload",
        {"configuration": q.tolist()},
        {"minimum_effective_mass_kg": float(values[0, 0]), "loaded_effective_mass_kg": float(values[-1, 0])},
        {"payload_kg": payload, "small_mass_kg": values[:, 0], "large_mass_kg": values[:, 1]},
        [
            curve(
                "Coupled Cartesian effective inertia",
                "payload_kg",
                [("small_mass_kg", "Small eigenvalue"), ("large_mass_kg", "Large eigenvalue")],
                "Payload (kg)",
                "Effective mass (kg)",
            )
        ],
        assumptions=[
            "Full generalized-force authority is a mechanics diagnostic, not a claim of full aircraft actuation."
        ],
    )


def velocity(scale=1, seed=42):
    from .planar import mass_matrix
    from .taskspace import minimum_energy_velocity, tip

    q = np.array([0, 1, 0.1, 0.4, -0.7])
    mass = mass_matrix(q)
    jac = tip(q)[1]
    angle = np.linspace(0, 2 * np.pi, 181 * scale)
    dynamic = []
    euclidean = []
    for value in angle:
        desired = 0.2 * np.array([np.cos(value), np.sin(value)])
        r = minimum_energy_velocity(mass, jac, desired)
        v = np.linalg.pinv(jac) @ desired
        dynamic.append(r["kinetic_j"])
        euclidean.append(0.5 * v @ mass @ v)
    return make(
        "dynamic-task-velocity",
        {"task_speed_m_s": 0.2},
        {"maximum_energy_saving_j": float(max(np.array(euclidean) - dynamic))},
        {
            "direction_deg": np.rad2deg(angle),
            "dynamic_j": np.array(dynamic),
            "euclidean_j": np.array(euclidean),
        },
        [
            curve(
                "Instantaneous kinetic cost",
                "direction_deg",
                [("dynamic_j", "Dynamically consistent"), ("euclidean_j", "Euclidean pseudoinverse")],
                "Task direction (deg)",
                "Kinetic energy (J)",
            )
        ],
        assumptions=["Instantaneous velocity allocation, not integrated dynamically feasible flight."],
    )


def nullspace(scale=1, seed=42):
    from .planar import mass_matrix
    from .taskspace import joint_limit_motion, tip

    q1 = np.linspace(0.1, 0.95, 101 * scale)
    rates = []
    leak = []
    descent = []
    lo = np.array([-2, -1, -2, -1, -1])
    hi = np.array([2, 3, 2, 1, 1])
    for angle in q1:
        q = np.array([0, 1, 0, angle, -0.5])
        jac = tip(q)[1]
        v = joint_limit_motion(q, mass_matrix(q), jac, lo, hi)
        rates.append(v[3])
        leak.append(np.linalg.norm(jac @ v))
        descent.append((1 / (q - lo) - 1 / (hi - q)) @ v)
    return make(
        "nullspace-joint-barrier",
        {"joint_upper_rad": 1},
        {"maximum_tip_leak_m_s": float(max(leak)), "minimum_barrier_descent": float(min(descent))},
        {"joint1_rad": q1, "avoidance_rate_rad_s": np.array(rates), "tip_leak_m_s": np.array(leak)},
        [
            curve(
                "Kinetic-metric joint barrier motion",
                "joint1_rad",
                [("avoidance_rate_rad_s", "Joint 1 null velocity")],
                "Joint position (rad)",
                "Rate (rad/s)",
            ),
            curve(
                "Task-preservation residual",
                "joint1_rad",
                [("tip_leak_m_s", "Tip velocity leakage")],
                "Joint position (rad)",
                "Velocity (m/s)",
                ylim=(-1e-10, 1e-10),
            ),
        ],
    )


def constraint(scale=1, seed=42):
    from .planar import gravity, mass_matrix
    from .taskspace import constrained_acceleration, tip

    angle = np.linspace(-1.2, 1.2, 101 * scale)
    reaction = []
    residual = []
    for value in angle:
        q = np.array([0, 1, 0.1, 0.4, value])
        jac = tip(q)[1]
        r = constrained_acceleration(mass_matrix(q), jac, np.zeros(5), gravity(q), np.zeros(2), np.zeros(2))
        reaction.append(r["reaction"])
        residual.append(np.linalg.norm(jac @ r["acceleration"]))
    reaction = np.array(reaction)
    return make(
        "bilateral-contact-reaction",
        {"desired_tip_acceleration_m_s2": [0, 0]},
        {"max_constraint_residual_m_s2": float(max(residual))},
        {"joint2_rad": angle, "horizontal_n": reaction[:, 0], "vertical_n": reaction[:, 1]},
        [
            curve(
                "Bilateral constraint reactions",
                "joint2_rad",
                [("horizontal_n", "Horizontal"), ("vertical_n", "Vertical")],
                "Joint 2 (rad)",
                "Constraint force (N)",
            )
        ],
        assumptions=[
            "Bilateral contact can pull or push. This is distinct from unilateral frictional impact."
        ],
    )


def work(scale=1, seed=42):
    from .core import stream
    from .taskspace import impedance

    rng = stream(seed, "virtual-work")
    n = 200 * scale
    task = []
    generalized = []
    for _ in range(n):
        q = np.array([0, 1, 0, 0.4, -0.7]) + rng.normal(0, 0.05, 5)
        v = rng.normal(0, 0.2, 5)
        r = impedance(q, v, [0.2, 0.5])
        task.append(r["task_power_w"])
        generalized.append(r["generalized_force"] @ v)
    residual = np.array(generalized) - task
    return make(
        "impedance-virtual-work",
        {"stiffness_n_m": 100, "damping_n_s_m": 10},
        {"max_power_defect_w": float(max(abs(residual)))},
        {
            "sample": np.arange(n),
            "task_w": np.array(task),
            "generalized_w": np.array(generalized),
            "residual_w": residual,
        },
        [
            curve(
                "Impedance power in two coordinate spaces",
                "sample",
                [("task_w", "Cartesian"), ("generalized_w", "Generalized")],
                "Independent motion sample",
                "Power (W)",
            ),
            curve(
                "Virtual-work closure",
                "sample",
                [("residual_w", "Power difference")],
                "Sample",
                "Residual (W)",
                ylim=(-1e-10, 1e-10),
            ),
        ],
    )


def authority(scale=1, seed=42):
    from .taskspace import scale_force, tip

    q = np.array([0, 1, 0, 0.4, -0.7])
    jac = tip(q)[1]
    angle = np.linspace(0, 2 * np.pi, 181 * scale)
    scales = []
    achieved = []
    for value in angle:
        r = scale_force(jac, 20 * np.array([np.cos(value), np.sin(value)]), [30, 30, 2, 1, 1])
        scales.append(r["scale"])
        achieved.append(np.linalg.norm(r["force"]))
    return make(
        "task-force-authority",
        {"requested_force_n": 20, "generalized_component_limits": [30, 30, 2, 1, 1]},
        {"minimum_scale": float(min(scales))},
        {"angle_deg": np.rad2deg(angle), "available_force_n": np.array(achieved), "scale": np.array(scales)},
        [
            curve(
                "Idealized generalized-limit force envelope",
                "angle_deg",
                [("available_force_n", "Scaled task force")],
                "Force direction (deg)",
                "Task force magnitude (N)",
            )
        ],
        assumptions=[
            "Component-box generalized limits are a design bound. They do not model the aircraft's underactuated thrust direction or replace rotor allocation."
        ],
    )


def retiming(scale=1, seed=42):
    from .motion import minimum_duration, quintic

    start = [0, 0]
    stop = [0.8, -1.2]
    duration = minimum_duration(start, stop, [1, 1], [2, 2], [10, 10])
    t = np.linspace(0, duration, 501 * scale)
    r = quintic(t, duration, start, stop)
    return make(
        "bounded-quintic-motion",
        {"velocity_limit_rad_s": 1, "acceleration_limit_rad_s2": 2, "jerk_limit_rad_s3": 10},
        {"duration_s": duration, "max_velocity_rad_s": float(max(abs(r["velocity"]).ravel()))},
        {
            "time_s": t,
            "joint1_rad": r["position"][:, 0],
            "joint2_rad": r["position"][:, 1],
            "joint1_rate": r["velocity"][:, 0],
            "joint2_rate": r["velocity"][:, 1],
            "joint1_accel": r["acceleration"][:, 0],
            "joint2_accel": r["acceleration"][:, 1],
        },
        [
            curve(
                "Synchronized rest-to-rest arm path",
                "time_s",
                [("joint1_rad", "Joint 1"), ("joint2_rad", "Joint 2")],
                "Time (s)",
                "Joint angle (rad)",
            ),
            curve(
                "Bounded joint rates",
                "time_s",
                [("joint1_rate", "Joint 1"), ("joint2_rate", "Joint 2")],
                "Time (s)",
                "Rate (rad/s)",
            ),
            curve(
                "Bounded joint acceleration",
                "time_s",
                [("joint1_accel", "Joint 1"), ("joint2_accel", "Joint 2")],
                "Time (s)",
                "Acceleration (rad/s^2)",
            ),
        ],
    )


def waypoints(scale=1, seed=42):
    from .motion import waypoints as evaluate

    t = np.linspace(0, 4, 801 * scale)
    r = evaluate(t, [[0, 0], [0.7, -0.4], [-0.2, 0.5]], [2, 2])
    return make(
        "rest-to-rest-waypoints",
        {"waypoints_rad": [[0, 0], [0.7, -0.4], [-0.2, 0.5]], "durations_s": [2, 2]},
        {"final_joint1_rad": float(r["position"][-1, 0])},
        {
            "time_s": t,
            "q1": r["position"][:, 0],
            "q2": r["position"][:, 1],
            "v1": r["velocity"][:, 0],
            "a1": r["acceleration"][:, 0],
        },
        [
            curve(
                "Two synchronized waypoint segments",
                "time_s",
                [("q1", "Joint 1"), ("q2", "Joint 2")],
                "Time (s)",
                "Angle (rad)",
            ),
            curve(
                "Rest at interior waypoint", "time_s", [("v1", "Joint 1 rate")], "Time (s)", "Rate (rad/s)"
            ),
            curve(
                "Continuous acceleration at the waypoint",
                "time_s",
                [("a1", "Joint 1 acceleration")],
                "Time (s)",
                "Acceleration (rad/s^2)",
            ),
        ],
        assumptions=["Piecewise rest-to-rest quintics are C2; jerk can jump at segment boundaries."],
    )


def clearance(scale=1, seed=42):
    from .motion import segment_clearance

    direct = segment_clearance([0, 0], [1, 1], [[0.5, 0.5]], [0.25])
    first = segment_clearance([0, 0], [0, 1], [[0.5, 0.5]], [0.25])
    last = segment_clearance([0, 1], [1, 1], [[0.5, 0.5]], [0.25])
    angle = np.linspace(0, 2 * np.pi, 181 * scale)
    arrays = {
        "direct_x": np.array([0.0, 1]),
        "direct_z": np.array([0.0, 1]),
        "detour_x": np.array([0.0, 0, 1]),
        "detour_z": np.array([0.0, 1, 1]),
        "obstacle_x": 0.5 + 0.25 * np.cos(angle),
        "obstacle_z": 0.5 + 0.25 * np.sin(angle),
    }
    plots = [
        curve("End-effector segment geometry", "direct_x", [("direct_z", "Direct")], "x (m)", "z (m)"),
        curve("Declared detour", "detour_x", [("detour_z", "Detour")], "x (m)", "z (m)"),
        curve("Obstacle boundary", "obstacle_x", [("obstacle_z", "Circular obstacle")], "x (m)", "z (m)"),
    ]
    for plot in plots:
        plot["aspect"] = "equal"
    return make(
        "continuous-path-clearance",
        {"obstacle_radius_m": 0.25},
        {
            "direct_collision": direct["collision"],
            "detour_clearance_m": float(min(first["clearance_m"][0], last["clearance_m"][0])),
        },
        arrays,
        plots,
        assumptions=[
            "Point end-effector line-segment geometry only; link swept volume and aircraft collision are not checked."
        ],
    )


def jerk(scale=1, seed=42):
    from .motion import jerk_cost, quintic

    duration = np.linspace(0.5, 3, 61 * scale)
    analytic = []
    numeric = []
    for value in duration:
        t = np.linspace(0, value, 1001)
        r = quintic(t, value, [0, 0], [0.8, -1.2])
        analytic.append(jerk_cost([0, 0], [0.8, -1.2], value))
        numeric.append(np.trapezoid(np.sum(r["jerk"] ** 2, axis=1), t))
    relative = np.max(abs(np.array(numeric) / analytic - 1))
    return make(
        "quintic-jerk-tradeoff",
        {"move_rad": [0.8, -1.2]},
        {"max_relative_quadrature_error": float(relative)},
        {"duration_s": duration, "analytic": np.array(analytic), "quadrature": np.array(numeric)},
        [
            curve(
                "Rest-to-rest smoothness cost",
                "duration_s",
                [("analytic", "Analytical"), ("quadrature", "Numerical quadrature")],
                "Duration (s)",
                "Integrated squared jerk (rad^2/s^5)",
                yscale="log",
            )
        ],
    )


def payload(scale=1, seed=42):
    from .core import stream
    from .identification import payload_fit, payload_regressor
    from .planar import Model, coriolis_matrix, gravity, mass_matrix

    n = 100 * scale
    t = np.linspace(0, 5, n)
    phi = []
    observed = []
    estimate = []
    uncertainty = []
    rng = stream(seed, "payload-fit")
    p = Model(payload_kg=0.35)
    for time in t:
        q = np.array([0, 1, 0.1 * np.sin(0.5 * time), 0.4 * np.sin(time), -0.6 + 0.2 * np.cos(time)])
        v = np.array([0, 0, 0.05 * np.cos(0.5 * time), 0.4 * np.cos(time), -0.2 * np.sin(time)])
        a = np.array([0, 0, -0.025 * np.sin(0.5 * time), -0.4 * np.sin(time), -0.2 * np.cos(time)])
        reg = payload_regressor(q, v, a)
        torque = mass_matrix(q, p) @ a + coriolis_matrix(q, v, p) @ v + gravity(q, p) + rng.normal(0, 0.01, 5)
        phi.extend(reg["column"][3:])
        observed.extend((torque - reg["baseline"])[3:])
        if len(phi) >= 4:
            fitted = payload_fit(phi, observed, 0.01)
            estimate.append(fitted["payload_kg"])
            uncertainty.append(fitted["standard_deviation_kg"])
        else:
            estimate.append(0.0)
            uncertainty.append(0.0)
    valid = np.arange(n) > 0
    return make(
        "payload-mass-identification",
        {"true_payload_kg": 0.35, "joint_torque_noise_nm": 0.01},
        {"estimated_payload_kg": estimate[-1], "declared_std_kg": uncertainty[-1]},
        {"time_s": t, "estimate_kg": np.array(estimate), "reference_kg": np.full(n, 0.35), "valid": valid},
        [
            curve(
                "Measured joint-reaction payload fit",
                "time_s",
                [("estimate_kg", "Estimate", "valid"), ("reference_kg", "Synthesis reference")],
                "Time (s)",
                "Payload mass (kg)",
            )
        ],
        assumptions=[
            "Configuration, velocity and acceleration observations are supplied; no derivative-noise model or flight estimator is claimed."
        ],
    )
