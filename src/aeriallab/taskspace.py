import numpy as np

from .core import finite


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
