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


def scale_force(jacobian, force, limits):
    jac = finite(jacobian, real=True, ndim=2)
    f = finite(force, real=True, ndim=1)
    bound = finite(limits, real=True, ndim=1)
    if jac.shape != (len(f), len(bound)) or np.any(bound <= 0):
        raise ValueError("positive generalized limits and aligned task force required")
    requested = jac.T @ f
    ratio = max(np.max(abs(requested) / bound), 1.0)
    return {"force": f / ratio, "generalized": requested / ratio, "scale": float(1 / ratio)}


def constrained_acceleration(mass, jacobian, applied_force, bias, desired_task_acceleration, convective):
    matrix = metric(mass)
    jac = finite(jacobian, real=True, ndim=2)
    force = finite(applied_force, real=True, ndim=1)
    b = finite(bias, real=True, ndim=1)
    desired = finite(desired_task_acceleration, real=True, ndim=1)
    drift = finite(convective, real=True, ndim=1)
    n = len(matrix)
    k = len(jac)
    if (
        jac.shape != (k, n)
        or force.shape != (n,)
        or b.shape != (n,)
        or desired.shape != (k,)
        or drift.shape != (k,)
    ):
        raise ValueError("aligned constrained mechanics required")
    solution = np.linalg.solve(
        np.block([[matrix, -jac.T], [jac, np.zeros((k, k))]]), np.r_[force - b, desired - drift]
    )
    return {"acceleration": solution[:n], "reaction": solution[n:]}


def center_of_mass(q, model=None):
    from .planar import Model, configuration, kinematics

    p = Model() if model is None else model
    x = configuration(q)
    kin = kinematics(x, p)
    position = p.base_mass_kg * x[:2]
    jac = p.base_mass_kg * np.eye(2, 5)
    for name, mass in (("link1", p.link1_mass_kg), ("link2", p.link2_mass_kg), ("tip", p.payload_kg)):
        position += mass * kin[name][0]
        jac += mass * kin[name][1]
    return {"position_m": position / p.total_mass, "jacobian": jac / p.total_mass}


def joint_limit_motion(q, mass, jacobian, lower, upper, gain=0.1):
    x = finite(q, real=True, ndim=1)
    lo = finite(lower, real=True, ndim=1)
    hi = finite(upper, real=True, ndim=1)
    if x.shape != lo.shape or x.shape != hi.shape or np.any(x <= lo) or np.any(x >= hi):
        raise ValueError("configuration strictly within declared bounds required")
    gradient = 1 / (x - lo) - 1 / (hi - x)
    return dynamic_inverse(mass, jacobian)["null"] @ np.linalg.solve(metric(mass), positive(gain) * gradient)
