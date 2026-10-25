import numpy as np


class Attitude:
    def __init__(self, angle_rad=0, gyro_std=0.003, bias_walk_std=0.0005):
        from .core import finite, positive

        self.state = np.array([float(finite(angle_rad, real=True)), 0.0])
        self.covariance = np.diag([0.02**2, 0.01**2])
        self.gyro_std = positive(gyro_std, allow_zero=True)
        self.bias_walk = positive(bias_walk_std, allow_zero=True)

    def predict(self, gyro_rad_s, dt_s):
        from .core import finite, positive

        gyro = float(finite(gyro_rad_s, real=True))
        dt = positive(dt_s)
        f = np.array([[1, -dt], [0, 1.0]])
        self.state[0] = (self.state[0] + (gyro - self.state[1]) * dt + np.pi) % (2 * np.pi) - np.pi
        self.covariance = f @ self.covariance @ f.T + np.diag(
            [self.gyro_std**2 * dt**2, self.bias_walk**2 * dt]
        )
        return self.state.copy()


def _attitude_correct(observer, angle_rad, std_rad=0.003, gate=9):
    from .core import finite, positive

    angle = float(finite(angle_rad, real=True))
    variance = positive(std_rad) ** 2
    y = (angle - observer.state[0] + np.pi) % (2 * np.pi) - np.pi
    p = observer.covariance
    s = p[0, 0] + variance
    nis = y * y / s
    if nis > positive(gate):
        return {"accepted": False, "nis": float(nis)}
    gain = p[:, 0] / s
    observer.state += gain * y
    observer.state[0] = (observer.state[0] + np.pi) % (2 * np.pi) - np.pi
    j = np.eye(2) - np.outer(gain, [1, 0])
    observer.covariance = j @ p @ j.T + np.outer(gain, gain) * variance
    observer.covariance = (observer.covariance + observer.covariance.T) / 2
    return {"accepted": True, "nis": float(nis)}


def correct_attitude(observer, angle_rad, std_rad=0.003, gate=9):
    return _attitude_correct(observer, angle_rad, std_rad, gate)


def navigation_predict(state, covariance, acceleration_world, dt_s, acceleration_std=0.1, bias_walk_std=0.01):
    from .core import finite, positive

    x = finite(state, real=True)
    p = finite(covariance, real=True)
    a = finite(acceleration_world, real=True)
    dt = positive(dt_s)
    std = positive(acceleration_std, allow_zero=True)
    walk = positive(bias_walk_std, allow_zero=True)
    if (
        x.shape != (6,)
        or p.shape != (6, 6)
        or a.shape != (2,)
        or not np.allclose(p, p.T)
        or np.min(np.linalg.eigvalsh(p)) < -1e-12
    ):
        raise ValueError("six-state PSD navigation estimate required")
    f = np.eye(6)
    f[:2, 2:4] = np.eye(2) * dt
    f[:2, 4:] = -np.eye(2) * dt**2 / 2
    f[2:4, 4:] = -np.eye(2) * dt
    b = np.vstack([np.eye(2) * dt**2 / 2, np.eye(2) * dt, np.zeros((2, 2))])
    q = b @ b.T * std**2
    q[4:, 4:] += np.eye(2) * walk**2 * dt
    return f @ x + b @ a, f @ p @ f.T + q


def navigation_correct(state, covariance, position_m, std_m=0.02, gate=9.21034):
    from .core import finite, positive

    x = finite(state, real=True)
    p = finite(covariance, real=True)
    z = finite(position_m, real=True)
    variance = positive(std_m) ** 2
    if x.shape != (6,) or p.shape != (6, 6) or z.shape != (2,):
        raise ValueError("navigation state and planar position measurement required")
    residual = z - x[:2]
    s = p[:2, :2] + np.eye(2) * variance
    nis = float(residual @ np.linalg.solve(s, residual))
    if nis > positive(gate):
        return x.copy(), p.copy(), False, nis
    gain = np.linalg.solve(s, p[:, :2].T).T
    h = np.column_stack([np.eye(2), np.zeros((2, 4))])
    j = np.eye(6) - gain @ h
    post = j @ p @ j.T + gain @ gain.T * variance
    return x + gain @ residual, (post + post.T) / 2, True, nis


class Navigation:
    def __init__(self, position=(0.03, 1.97)):
        from .core import finite

        p = finite(position, real=True)
        if p.shape != (2,):
            raise ValueError("surveyed initial planar position required")
        self.attitude = Attitude()
        self.state = np.r_[p, np.zeros(4)]
        self.covariance = np.diag([0.1**2, 0.1**2, 0.2**2, 0.2**2, 0.1**2, 0.1**2])
        self.last_acceleration = np.zeros(2)

    def update(self, imu, dt_s, position=None, attitude_reference=None, gravity=9.80665):
        from .core import positive
        from .sensors import rotation

        positive(gravity, allow_zero=True)
        self.attitude.predict(imu["gyro_rad_s"], dt_s)
        attitude_result = {"accepted": False, "nis": 0.0}
        if attitude_reference is not None:
            attitude_result = correct_attitude(self.attitude, attitude_reference)
        if imu["valid"]:
            self.last_acceleration = rotation(self.attitude.state[0]) @ imu["specific_force_m_s2"] - np.array(
                [0, gravity]
            )
        self.state, self.covariance = navigation_predict(
            self.state,
            self.covariance,
            self.last_acceleration,
            dt_s,
            acceleration_std=0.1 if imu["valid"] else 5,
        )
        accepted = False
        nis = 0.0
        if position is not None:
            self.state, self.covariance, accepted, nis = navigation_correct(
                self.state, self.covariance, position
            )
        return {
            "position_accepted": accepted,
            "position_nis": nis,
            "attitude_accepted": attitude_result["accepted"],
            "imu_valid": bool(imu["valid"]),
        }


def navigation_observability(times):
    from .core import timebase

    t = timebase(times)
    h = np.column_stack([np.eye(2), np.zeros((2, 4))])
    matrices = []
    for elapsed in t - t[0]:
        f = np.eye(6)
        f[:2, 2:4] = np.eye(2) * elapsed
        f[:2, 4:] = -np.eye(2) * elapsed**2 / 2
        f[2:4, 4:] = -np.eye(2) * elapsed
        matrices.append(h @ f)
    singular = np.linalg.svd(np.vstack(matrices), compute_uv=False)
    return {"rank": int(np.linalg.matrix_rank(np.vstack(matrices))), "singular_values": singular}
