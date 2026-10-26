import numpy as np

from .core import finite, positive


def payload_regressor(q, velocity, acceleration, model=None):
    from dataclasses import replace

    from .planar import Model, configuration, coriolis_matrix, gravity, mass_matrix

    p = Model() if model is None else model
    x = configuration(q)
    v = configuration(velocity)
    a = configuration(acceleration)

    def inverse(mass):
        candidate = replace(p, payload_kg=mass)
        return mass_matrix(x, candidate) @ a + coriolis_matrix(x, v, candidate) @ v + gravity(x, candidate)

    baseline = inverse(0)
    return {"baseline": baseline, "column": inverse(1) - baseline}


def payload_fit(regressor, measured_minus_baseline, noise_std):
    phi = finite(regressor, real=True, ndim=1)
    y = finite(measured_minus_baseline, real=True, ndim=1)
    sigma = np.broadcast_to(finite(noise_std, real=True), phi.shape)
    if phi.shape != y.shape or len(phi) < 3 or np.any(sigma <= 0):
        raise ValueError("aligned reaction samples and positive uncertainties required")
    a = phi / sigma
    b = y / sigma
    information = a @ a
    if information < 1e-8:
        raise ValueError("payload excitation is insufficient")
    unconstrained = float(a @ b / information)
    estimate = max(0.0, unconstrained)
    return {
        "payload_kg": estimate,
        "standard_deviation_kg": float(1 / np.sqrt(information)),
        "at_boundary": unconstrained < 0,
        "weighted_residual": b - estimate * a,
    }


def damping_fit(velocities, residual_forces):
    v = finite(velocities, real=True, ndim=2)
    force = finite(residual_forces, real=True, ndim=2)
    if v.shape != force.shape or len(v) < 3:
        raise ValueError("aligned residual force and velocity samples required")
    excitation = np.sum(v * v, axis=0)
    if np.any(excitation < 1e-8):
        raise ValueError("every damping channel must be excited")
    coefficient = np.sum(v * force, axis=0) / excitation
    if np.any(coefficient < 0):
        raise ValueError("fitted damping violates passivity")
    return {"damping": coefficient, "residual": force - v * coefficient}


def wrench_fit(jacobian, generalized_reaction, noise_std):
    jac = finite(jacobian, real=True, ndim=2)
    force = finite(generalized_reaction, real=True, ndim=1)
    sigma = np.broadcast_to(finite(noise_std, real=True), force.shape)
    if jac.shape[1] != len(force) or np.any(sigma <= 0):
        raise ValueError("aligned reaction channels and uncertainties required")
    design = jac.T / sigma[:, None]
    information = design.T @ design
    if np.linalg.cond(information) > 1e10:
        raise ValueError("contact wrench is unobservable")
    cov = np.linalg.inv(information)
    return {"force": cov @ design.T @ (force / sigma), "covariance": cov}


def friction_projection(force, coefficient=0.5):
    f = finite(force, real=True)
    mu = positive(coefficient, allow_zero=True)
    if f.shape != (2,):
        raise ValueError("tangential and positive-normal force required")
    tangent, normal = f
    if normal >= 0 and abs(tangent) <= mu * normal:
        return f.copy()
    if mu == 0:
        return np.array([0.0, max(0.0, normal)])
    candidate = max(0.0, (mu * abs(tangent) + normal) / (1 + mu * mu))
    return np.array([np.sign(tangent) * mu * candidate, candidate])


def frictional_impact(q, velocity, coefficient=0.5, model=None):
    from .planar import configuration, kinematics, mass_matrix

    x = configuration(q)
    v = configuration(velocity)
    mu = positive(coefficient, allow_zero=True)
    mass = mass_matrix(x, model)
    jac = kinematics(x, model)["tip"][1]
    contact = jac @ v
    if contact[1] >= 0:
        return {"velocity": v.copy(), "impulse": np.zeros(2), "dissipated_j": 0.0, "mode": "separating"}
    inverse = np.linalg.solve(mass, jac.T)
    w = jac @ inverse
    candidates = []
    stick = -np.linalg.solve(w, contact)
    if stick[1] >= 0 and abs(stick[0]) <= mu * stick[1] + 1e-12:
        candidates.append((stick, "sticking"))
    for sign in (-1, 1):
        denominator = w[1, 1] + sign * mu * w[1, 0]
        if denominator <= 0:
            continue
        normal = -contact[1] / denominator
        impulse = np.array([sign * mu * normal, normal])
        after = contact + w @ impulse
        if normal >= 0 and sign * after[0] <= 1e-10:
            candidates.append((impulse, "sliding"))
    if not candidates:
        raise ValueError("no consistent rigid Coulomb impact branch")
    impulse, mode = min(
        candidates, key=lambda item: float((v + inverse @ item[0]) @ mass @ (v + inverse @ item[0]))
    )
    out = v + inverse @ impulse
    loss = float(0.5 * v @ mass @ v - 0.5 * out @ mass @ out)
    if loss < -1e-9:
        raise ValueError("impact branch generates unsupported energy")
    return {"velocity": out, "impulse": impulse, "dissipated_j": max(0.0, loss), "mode": mode}
