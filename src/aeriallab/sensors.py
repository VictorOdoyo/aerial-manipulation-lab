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


class Suite:
    def __init__(
        self,
        seed=42,
        gyro_bias=0.005,
        accel_bias=(0.03, -0.02),
        gyro_std=0.003,
        accel_std=0.05,
        position_std=0.02,
        attitude_std=0.003,
    ):
        from .core import finite, positive, stream

        self.bias = float(finite(gyro_bias, real=True))
        self.accel_bias = finite(accel_bias, real=True)
        if self.accel_bias.shape != (2,):
            raise ValueError("planar accelerometer bias required")
        self.std = {
            key: positive(value, allow_zero=True)
            for key, value in (
                ("gyro", gyro_std),
                ("accel", accel_std),
                ("position", position_std),
                ("attitude", attitude_std),
            )
        }
        self.rng = {key: stream(seed, "sensor-" + key) for key in [*self.std, "rotor"]}

    def imu(self, pitch, rate, acceleration_world, gravity=9.80665):
        from .core import finite

        finite([pitch, rate], real=True)
        f = (
            specific_force(acceleration_world, pitch, gravity)
            + self.accel_bias
            + self.rng["accel"].normal(0, self.std["accel"], 2)
        )
        gyro = rate + self.bias + self.rng["gyro"].normal(0, self.std["gyro"])
        valid = bool(np.all(abs(f) < 16 * 9.80665))
        return {
            "specific_force_m_s2": np.clip(f, -16 * 9.80665, 16 * 9.80665),
            "gyro_rad_s": float(gyro),
            "valid": valid,
        }

    def position(self, position):
        from .core import finite

        p = finite(position, real=True)
        if p.shape != (2,):
            raise ValueError("planar position required")
        return p + self.rng["position"].normal(0, self.std["position"], 2)

    def attitude(self, pitch):
        return float(
            (pitch + self.rng["attitude"].normal(0, self.std["attitude"]) + np.pi) % (2 * np.pi) - np.pi
        )


def rotor_tachometer(suite, speed_rad_s, std_rad_s=1):
    from .core import finite, positive

    speed = finite(speed_rad_s, real=True)
    if speed.shape != (2,) or np.any(speed < 0):
        raise ValueError("two nonnegative rotor-pair speeds required")
    return np.maximum(0, speed + suite.rng["rotor"].normal(0, positive(std_rad_s, allow_zero=True), 2))
