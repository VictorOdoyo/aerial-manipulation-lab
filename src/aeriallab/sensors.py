import numpy as np


def rotation(pitch_rad):
    if not np.isfinite(pitch_rad):
        raise ValueError("finite pitch required")
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    return np.array([[c, -s], [s, c]])


def specific_force(acceleration_world, pitch_rad, gravity=9.80665):
    from .core import finite, positive

    a = finite(acceleration_world, real=True)
    if a.shape != (2,):
        raise ValueError("planar world acceleration required")
    return rotation(pitch_rad).T @ (a + np.array([0, positive(gravity, allow_zero=True)]))


def encoder(joints_rad, counts=16384):
    from .core import finite, integer

    q = finite(joints_rad, real=True)
    integer(counts, 4)
    if q.shape != (2,):
        raise ValueError("two joint encoder angles required")
    codes = np.rint(q * counts / (2 * np.pi)).astype(np.int64) % counts
    angle = (codes * 2 * np.pi / counts + np.pi) % (2 * np.pi) - np.pi
    return {"codes": codes, "angle_rad": angle, "lsb_rad": 2 * np.pi / counts}


class EncoderVelocity:
    def __init__(self, time_constant_s=0.03):
        from .core import positive

        self.tau = positive(time_constant_s)
        self.previous = None
        self.velocity = np.zeros(2)

    def update(self, angles, dt_s):
        from .core import finite, positive

        q = finite(angles, real=True)
        dt = positive(dt_s)
        if q.shape != (2,):
            raise ValueError("two joint angles required")
        if self.previous is not None:
            delta = (q - self.previous + np.pi) % (2 * np.pi) - np.pi
            raw = delta / dt
            decay = np.exp(-dt / self.tau)
            self.velocity = decay * self.velocity + (1 - decay) * raw
        self.previous = q.copy()
        return self.velocity.copy()
