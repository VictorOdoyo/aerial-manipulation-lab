from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Model:
    base_mass_kg: float = 1.8
    base_inertia_kg_m2: float = 0.035
    link1_mass_kg: float = 0.25
    link2_mass_kg: float = 0.18
    link1_m: float = 0.35
    link2_m: float = 0.3
    payload_kg: float = 0.1
    gravity_m_s2: float = 9.80665

    def __post_init__(self):
        from .core import positive

        for key in (
            "base_mass_kg",
            "base_inertia_kg_m2",
            "link1_mass_kg",
            "link2_mass_kg",
            "link1_m",
            "link2_m",
        ):
            positive(getattr(self, key), key)
        positive(self.payload_kg, allow_zero=True)
        positive(self.gravity_m_s2, allow_zero=True)

    @property
    def total_mass(self):
        return self.base_mass_kg + self.link1_mass_kg + self.link2_mass_kg + self.payload_kg

    @property
    def inertias(self):
        return self.link1_mass_kg * self.link1_m**2 / 12, self.link2_mass_kg * self.link2_m**2 / 12


def configuration(q):
    from .core import finite

    x = finite(q, real=True)
    if x.shape != (5,):
        raise ValueError("x z pitch relative-joint1 relative-joint2 coordinates required")
    return x


def _kinematics(q, p):
    x, z, theta, q1, q2 = q
    a = theta + q1
    b = a + q2
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    dtype = np.result_type(q, float)

    def point(first, second):
        position = np.array([x + first * sa + second * sb, z - first * ca - second * cb], dtype=dtype)
        dx = first * ca + second * cb
        dz = first * sa + second * sb
        jac = np.array([[1, 0, dx, dx, second * cb], [0, 1, dz, dz, second * sb]], dtype=dtype)
        return position, jac

    return {
        "link1": point(p.link1_m / 2, 0),
        "link2": point(p.link1_m, p.link2_m / 2),
        "tip": point(p.link1_m, p.link2_m),
    }


def kinematics(q, model=None):
    p = Model() if model is None else model
    return _kinematics(configuration(q), p)


def _mass(q, p):
    kin = _kinematics(q, p)
    matrix = np.diag(
        np.array([p.base_mass_kg, p.base_mass_kg, p.base_inertia_kg_m2, 0, 0], dtype=np.result_type(q, float))
    )
    for key, mass in (("link1", p.link1_mass_kg), ("link2", p.link2_mass_kg), ("tip", p.payload_kg)):
        jac = kin[key][1]
        matrix += mass * jac.T @ jac
    w1 = np.array([0, 0, 1, 1, 0])
    w2 = np.array([0, 0, 1, 1, 1])
    i1, i2 = p.inertias
    return matrix + i1 * np.outer(w1, w1) + i2 * np.outer(w2, w2)


def mass_matrix(q, model=None):
    p = Model() if model is None else model
    return _mass(configuration(q), p)


def potential(q, model=None):
    p = Model() if model is None else model
    x = configuration(q)
    kin = _kinematics(x, p)
    return float(
        p.gravity_m_s2
        * (
            p.base_mass_kg * x[1]
            + p.link1_mass_kg * kin["link1"][0][1]
            + p.link2_mass_kg * kin["link2"][0][1]
            + p.payload_kg * kin["tip"][0][1]
        )
    )


def gravity(q, model=None):
    p = Model() if model is None else model
    x = configuration(q)
    kin = _kinematics(x, p)
    force = np.array([0, p.base_mass_kg * p.gravity_m_s2, 0, 0, 0])
    for key, mass in (("link1", p.link1_mass_kg), ("link2", p.link2_mass_kg), ("tip", p.payload_kg)):
        force += mass * p.gravity_m_s2 * kin[key][1][1]
    return force


def mass_derivatives(q, model=None):
    p = Model() if model is None else model
    x = configuration(q)
    h = 1e-30
    return np.stack(
        [np.imag(_mass(x.astype(complex) + 1j * h * np.eye(5)[k], p)) / h for k in range(5)], axis=2
    )


def coriolis_matrix(q, velocity, model=None):
    v = configuration(velocity)
    derivatives = mass_derivatives(q, model)
    gamma = 0.5 * (derivatives + derivatives.transpose(0, 2, 1) - derivatives.transpose(2, 0, 1))
    return np.einsum("ijk,k->ij", gamma, v)


def generalized_force(
    q, thrust_n, body_torque_nm=0, joint_torques=(0, 0), wind_n=(0, 0), tip_force_n=(0, 0), model=None
):
    from .core import finite, positive

    p = Model() if model is None else model
    x = configuration(q)
    thrust = positive(thrust_n, allow_zero=True)
    joints = finite(joint_torques, real=True)
    wind = finite(wind_n, real=True)
    tip = finite(tip_force_n, real=True)
    if joints.shape != (2,) or wind.shape != (2,) or tip.shape != (2,) or not np.isfinite(body_torque_nm):
        raise ValueError("finite planar actuator wrenches required")
    force = np.array(
        [
            -thrust * np.sin(x[2]) + wind[0],
            thrust * np.cos(x[2]) + wind[1],
            body_torque_nm,
            joints[0],
            joints[1],
        ]
    )
    return force + kinematics(x, p)["tip"][1].T @ tip


def acceleration(q, velocity, force, model=None, damping=(0, 0, 0, 0, 0)):
    from .core import finite

    p = Model() if model is None else model
    x = configuration(q)
    v = configuration(velocity)
    u = configuration(force)
    d = finite(damping, real=True)
    if d.shape != (5,) or np.any(d < 0):
        raise ValueError("five nonnegative damping coefficients required")
    return np.linalg.solve(mass_matrix(x, p), u - coriolis_matrix(x, v, p) @ v - gravity(x, p) - d * v)


def rhs(state, force, model=None, damping=(0, 0, 0, 0, 0)):
    from .core import finite

    x = finite(state, real=True)
    if x.shape != (10,):
        raise ValueError("five positions and five generalized velocities required")
    return np.r_[x[5:], acceleration(x[:5], x[5:], force, model, damping)]


def step(state, force, dt_s, model=None, damping=(0, 0, 0, 0, 0)):
    from .core import finite, positive

    x = finite(state, real=True)
    dt = positive(dt_s)
    k1 = rhs(x, force, model, damping)
    k2 = rhs(x + dt * k1 / 2, force, model, damping)
    k3 = rhs(x + dt * k2 / 2, force, model, damping)
    k4 = rhs(x + dt * k3, force, model, damping)
    result = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite coupled state integration")
    return result


def energy(state, model=None):
    from .core import finite

    p = Model() if model is None else model
    x = finite(state, real=True)
    if x.shape != (10,):
        raise ValueError("ten-state energy required")
    kinetic = float(0.5 * x[5:] @ mass_matrix(x[:5], p) @ x[5:])
    potential_j = potential(x[:5], p)
    return {"kinetic_j": kinetic, "potential_j": potential_j, "total_j": kinetic + potential_j}


def actuator_power(velocity, force):
    return float(configuration(velocity) @ configuration(force))


def center_of_mass(q, model=None):
    p = Model() if model is None else model
    x = configuration(q)
    kin = kinematics(x, p)
    position = p.base_mass_kg * x[:2]
    jac = p.base_mass_kg * np.column_stack([np.eye(2), np.zeros((2, 3))])
    for key, mass in (("link1", p.link1_mass_kg), ("link2", p.link2_mass_kg), ("tip", p.payload_kg)):
        position = position + mass * kin[key][0]
        jac = jac + mass * kin[key][1]
    return position / p.total_mass, jac / p.total_mass


def linear_momentum(q, velocity, model=None):
    p = Model() if model is None else model
    return p.total_mass * center_of_mass(q, p)[1] @ configuration(velocity)
