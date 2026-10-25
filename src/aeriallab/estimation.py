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
