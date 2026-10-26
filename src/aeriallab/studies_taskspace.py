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
