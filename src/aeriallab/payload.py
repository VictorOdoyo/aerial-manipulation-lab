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


def release(state, payload_kg, model=None):
    from .core import finite, positive
    from .planar import Model, energy, kinematics, tip_velocity

    p = Model() if model is None else model
    x = finite(state, real=True)
    mass = positive(payload_kg)
    if x.shape != (10,) or mass > p.payload_kg:
        raise ValueError("available payload mass and ten-state aircraft required")
    new = replace(p, payload_kg=p.payload_kg - mass)
    velocity = tip_velocity(x[:5], x[5:], p)
    position = kinematics(x[:5], p)["tip"][0]
    object_energy = 0.5 * mass * (velocity @ velocity) + mass * p.gravity_m_s2 * position[1]
    return {
        "state": x.copy(),
        "model": new,
        "object_position_m": position,
        "object_velocity_m_s": velocity,
        "energy_defect_j": float(energy(x, p)["total_j"] - energy(x, new)["total_j"] - object_energy),
    }
