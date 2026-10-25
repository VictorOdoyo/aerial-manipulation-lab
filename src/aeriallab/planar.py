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
