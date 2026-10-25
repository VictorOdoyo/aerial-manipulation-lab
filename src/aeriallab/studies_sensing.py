import numpy as np

from .results import curve, make


def outage(scale=1, seed=42):
    from .estimation import Navigation
    from .sensors import Suite

    dt = 0.005 / scale
    t = np.arange(round(5 / dt) + 1) * dt
    suite = Suite(seed)
    nav = Navigation([0, 2])
    history = [nav.state.copy()]
    covariance = [nav.covariance.copy()]
    available = [False]
    accepted = [False]
    for k, time in enumerate(t[1:], 1):
        pitch = 0.1 * np.sin(0.7 * time)
        rate = 0.07 * np.cos(0.7 * time)
        acc = np.array([-0.05 * np.sin(0.5 * time), 0.016 * np.cos(0.4 * time)])
        position = np.array([0.2 * np.sin(0.5 * time), 2 + 0.1 * (1 - np.cos(0.4 * time))])
        imu = suite.imu(pitch, rate, acc)
        has_position = k % (20 * scale) == 0 and not 2 <= time <= 4
        measurement = suite.position(position) if has_position else None
        attitude = suite.attitude(pitch) if k % (4 * scale) == 0 else None
        result = nav.update(imu, dt, measurement, attitude)
        history.append(nav.state.copy())
        covariance.append(nav.covariance.copy())
        available.append(has_position)
        accepted.append(result["position_accepted"])
    history = np.array(history)
    covariance = np.array(covariance)
    truth = np.column_stack([0.2 * np.sin(0.5 * t), 2 + 0.1 * (1 - np.cos(0.4 * t))])
    error = np.linalg.norm(history[:, :2] - truth, axis=1)
    a = {
        "time_s": t,
        "truth_x_m": truth[:, 0],
        "truth_z_m": truth[:, 1],
        "estimate_x_m": history[:, 0],
        "estimate_z_m": history[:, 1],
        "position_error_m": error,
        "position_sigma_m": np.sqrt(np.trace(covariance[:, :2, :2], axis1=1, axis2=2)),
        "position_available": np.array(available, bool),
        "position_accepted": np.array(accepted, bool),
        "covariance": covariance,
        "state": history,
    }
    return make(
        "inertial-position-outage",
        {
            "seed": seed,
            "dt_s": dt,
            "outage_s": [2, 4],
            "position_period_s": 0.1,
            "attitude_period_s": 0.02,
            "sensor_model": {
                "gyro_bias_rad_s": 0.005,
                "accel_bias_body_m_s2": [0.03, -0.02],
                "gyro_std_rad_s": 0.003,
                "accel_std_m_s2": 0.05,
                "position_std_m": 0.02,
                "attitude_std_rad": 0.003,
            },
            "navigation": {
                "acceleration_sample_std": 0.1,
                "bias_walk_std": 0.01,
                "position_gate_nis": 9.21034,
                "attitude_gate_nis": 9,
            },
        },
        {"maximum_position_error_m": float(max(error)), "final_position_error_m": float(error[-1])},
        a,
        [
            curve(
                "Sensor-derived position",
                "time_s",
                [
                    ("truth_x_m", "Truth x"),
                    ("estimate_x_m", "Estimated x"),
                    ("truth_z_m", "Truth z"),
                    ("estimate_z_m", "Estimated z"),
                ],
                "Time (s)",
                "Position (m)",
            ),
            curve(
                "Error and predicted position uncertainty",
                "time_s",
                [
                    ("position_error_m", "Actual error norm"),
                    ("position_sigma_m", "Sqrt trace of position covariance"),
                ],
                "Time (s)",
                "Position scale (m)",
            ),
        ],
        assumptions=[
            "Prescribed synthetic motion generates sensors; the observer receives IMU and available noisy reference measurements only.",
            "World acceleration bias is a random-walk approximation to rotating body sensor bias; no calibrated confidence guarantee is claimed.",
        ],
    )


def observability(scale=1, seed=42):
    from .estimation import navigation_observability

    interval = np.geomspace(0.01, 1, 81 * scale)
    two = []
    three = []
    minimum = []
    for dt in interval:
        a = navigation_observability([0, float(dt)])
        b = navigation_observability([0, float(dt), float(2 * dt)])
        two.append(a["rank"])
        three.append(b["rank"])
        minimum.append(b["singular_values"][-1])
    arrays = {"interval_s": interval, "rank_two": two, "rank_three": three, "minimum_three": minimum}
    return make(
        "navigation-observability",
        {
            "state_units": ["m", "m", "m/s", "m/s", "m/s2", "m/s2"],
            "coordinate_scaling": "one unit of each stated SI coordinate",
        },
        {"two_measurement_rank": int(min(two)), "three_measurement_rank": int(min(three))},
        arrays,
        [
            curve(
                "Finite-horizon observable state count",
                "interval_s",
                [("rank_two", "Two position observations"), ("rank_three", "Three position observations")],
                "Observation interval (s)",
                "Observable rank",
                xscale="log",
                ylim=(0, 6.5),
            ),
            curve(
                "Weakest observable direction under declared scaling",
                "interval_s",
                [("minimum_three", "Three-observation smallest singular value")],
                "Observation interval (s)",
                "Dimensionless scaled singular value",
                xscale="log",
                yscale="log",
            ),
        ],
        assumptions=[
            "Linear held-input navigation model; coordinate scaling is explicit because observability singular values depend on state units."
        ],
    )
