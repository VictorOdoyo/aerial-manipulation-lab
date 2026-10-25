from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    duration_s: float = 6.0
    dt_s: float = 0.005
    seed: int = 42
    moving_base: bool = False
    moving_arm: bool = True
    reaction_compensation: bool = True
    feedback: str = "estimated"
    wind_peak_m_s: float = 0.0
    payload_kg: float = 0.1
    assumed_payload_kg: float = 0.1
    position_outage: tuple | None = None
    position_outlier: bool = False
    actuator_fault: bool = False
    fault_time_s: float = 3.0
    health_after: tuple = (0.2, 1.0)
    initial_soc: float = 0.9
    integral: bool = True

    def __post_init__(self):
        from .core import finite, integer, positive

        positive(self.duration_s)
        positive(self.dt_s)
        integer(self.seed, 0)
        positive(self.payload_kg, allow_zero=True)
        positive(self.assumed_payload_kg, allow_zero=True)
        positive(self.fault_time_s, allow_zero=True)
        if (
            self.dt_s > 0.01
            or self.duration_s < 2 * self.dt_s
            or self.seed >= 2**32
            or not 0 < self.initial_soc <= 1
        ):
            raise ValueError("resolved flight timing interior pack state and 32-bit seed required")
        if self.feedback not in ("estimated", "ideal"):
            raise ValueError("estimated or explicitly ideal feedback required")
        for key in (
            "moving_base",
            "moving_arm",
            "reaction_compensation",
            "position_outlier",
            "actuator_fault",
            "integral",
        ):
            if type(getattr(self, key)) is not bool:
                raise ValueError("flight switches must be booleans")
        finite(self.wind_peak_m_s, real=True)
        h = finite(self.health_after, real=True)
        if h.shape != (2,) or np.any((h < 0) | (h > 1)):
            raise ValueError("two bounded actuator health fractions required")
        if self.position_outage is not None:
            outage = finite(self.position_outage, real=True)
            if outage.shape != (2,) or not 0 <= outage[0] < outage[1]:
                raise ValueError("ordered nonnegative position outage required")

    def to_dict(self):
        return asdict(self)
