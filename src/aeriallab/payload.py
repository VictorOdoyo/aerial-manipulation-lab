from dataclasses import replace

import numpy as np


def pickup(state, payload_kg, object_velocity=(0, 0), model=None):
    from .core import finite, positive
    from .planar import Model, configuration, energy, kinematics, mass_matrix

    p = Model() if model is None else model
    x = finite(state, real=True)
    added = positive(payload_kg)
    velocity = finite(object_velocity, real=True)
    if x.shape != (10,) or velocity.shape != (2,):
        raise ValueError("ten-state aircraft and planar object velocity required")
    q = configuration(x[:5])
    jac = kinematics(q, p)["tip"][1]
    new = replace(p, payload_kg=p.payload_kg + added)
    generalized = mass_matrix(q, p) @ x[5:] + added * jac.T @ velocity
    out = x.copy()
    out[5:] = np.linalg.solve(mass_matrix(q, new), generalized)
    before = (
        energy(x, p)["total_j"]
        + 0.5 * added * (velocity @ velocity)
        + added * p.gravity_m_s2 * kinematics(q, p)["tip"][0][1]
    )
    after = energy(out, new)["total_j"]
    return {"state": out, "model": new, "dissipated_j": float(before - after)}
