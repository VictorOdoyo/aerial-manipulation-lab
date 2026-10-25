from dataclasses import dataclass


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
