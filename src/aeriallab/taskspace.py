import numpy as np

from .core import finite, positive


def metric(matrix):
    mass = finite(matrix, real=True, ndim=2)
    if mass.shape[0] != mass.shape[1] or not np.allclose(mass, mass.T) or min(np.linalg.eigvalsh(mass)) <= 0:
        raise ValueError("symmetric positive kinetic metric required")
    return mass


def tip(q, model=None):
    from .planar import kinematics

    return kinematics(q, model)["tip"]


def operational_inertia(mass, jacobian):
    matrix = metric(mass)
    jac = finite(jacobian, real=True, ndim=2)
    if jac.shape[1] != len(matrix):
        raise ValueError("task Jacobian must span generalized coordinates")
    inverse_times = np.linalg.solve(matrix, jac.T)
    inverse_task = jac @ inverse_times
    if np.linalg.matrix_rank(inverse_task) < len(jac):
        raise ValueError("task directions must be independent")
    return np.linalg.inv(inverse_task)


def dynamic_inverse(mass, jacobian):
    matrix = metric(mass)
    jac = finite(jacobian, real=True, ndim=2)
    task = operational_inertia(matrix, jac)
    inverse = np.linalg.solve(matrix, jac.T) @ task
    null = np.eye(len(matrix)) - inverse @ jac
    return {"inverse": inverse, "null": null, "task_inertia": task}


def minimum_energy_velocity(mass, jacobian, task_velocity):
    desired = finite(task_velocity, real=True, ndim=1)
    jac = finite(jacobian, real=True, ndim=2)
    if desired.shape != (len(jac),):
        raise ValueError("one task velocity per Jacobian row required")
    solution = dynamic_inverse(mass, jac)["inverse"] @ desired
    return {
        "generalized_velocity": solution,
        "kinetic_j": float(0.5 * solution @ metric(mass) @ solution),
        "achieved": jac @ solution,
    }


def convective_acceleration(q, velocity, model=None, step_s=1e-5):
    from .planar import configuration

    x = configuration(q)
    v = configuration(velocity)
    h = positive(step_s)
    return ((tip(x + h * v, model)[1] - tip(x - h * v, model)[1]) / (2 * h)) @ v


def impedance(q, velocity, target, stiffness_n_m=100, damping_n_s_m=10, model=None):
    from .planar import configuration

    x = configuration(q)
    v = configuration(velocity)
    desired = finite(target, real=True)
    if desired.shape != (2,):
        raise ValueError("planar Cartesian target required")
    point, jac = tip(x, model)
    force = positive(stiffness_n_m) * (desired - point) - positive(damping_n_s_m, allow_zero=True) * (jac @ v)
    return {"force_n": force, "generalized_force": jac.T @ force, "task_power_w": float(force @ (jac @ v))}
