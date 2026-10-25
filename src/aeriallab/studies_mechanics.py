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
