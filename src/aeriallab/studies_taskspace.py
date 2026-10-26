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
