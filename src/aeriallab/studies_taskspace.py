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
