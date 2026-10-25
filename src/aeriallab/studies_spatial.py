from dataclasses import asdict

import numpy as np

from .results import curve, make


def free_rotation(scale=1, seed=42):
    from .quadrotor import Model, angular_momentum, energy, step

    p = Model(gravity=0)
    dt = 0.005 / scale
    t = np.arange(round(1 / dt) + 1) * dt
    x = np.r_[np.zeros(6), [1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
    history = []
    energies = []
    momenta = []
    for _ in t:
        history.append(x.copy())
        energies.append(energy(x, p))
        momenta.append(angular_momentum(x, p))
        x = step(x, np.zeros(4), dt, model=p)
    history = np.array(history)
    momenta = np.array(momenta)
    a = {
        "time_s": t,
        "state": history,
        "omega_x": history[:, 10],
        "omega_y": history[:, 11],
        "omega_z": history[:, 12],
        "energy_j": energies,
        "quaternion_norm": np.linalg.norm(history[:, 6:10], axis=1),
        "inertial_momentum": momenta,
    }
    return make(
        "spatial-free-rotation",
        {"model": asdict(p), "dt_s": dt, "initial_state": history[0].tolist()},
        {
            "maximum_energy_defect_j": float(max(abs(np.array(energies) - energies[0]))),
            "maximum_momentum_defect": float(max(np.linalg.norm(momenta - momenta[0], axis=1))),
        },
        a,
        [
            curve(
                "Torque-free body-frame precession",
                "time_s",
                [("omega_x", "x"), ("omega_y", "y"), ("omega_z", "z")],
                "Time (s)",
                "Body rate (rad/s)",
            ),
            curve(
                "Physical kinetic-energy scale",
                "time_s",
                [("energy_j", "Rigid-body kinetic energy")],
                "Time (s)",
                "Energy (J)",
                ylim=(0, energies[0] * 1.1),
            ),
        ],
        assumptions=[
            "Separate full spatial rigid-body bench; the two-link arm coupling is planar in the main replay."
        ],
    )


def attitude_recovery(scale=1, seed=42):
    from .control import attitude_torque
    from .quadrotor import Model, allocate, step
    from .rotation import error, integrate, matrix

    p = Model()
    dt = 0.005 / scale
    t = np.arange(round(2 / dt) + 1) * dt
    x = np.r_[[0.0, 0.0, 2.0], np.zeros(3), integrate([1, 0, 0, 0], [0.25, -0.15, 0.05], 1), np.zeros(3)]
    states = []
    errors = []
    force_residual = []
    moment_residual = []
    for _ in t:
        torque = attitude_torque(x[6:10], x[10:], [1, 0, 0, 0], model=p)
        thrust = p.mass_kg * p.gravity / max(0.2, matrix(x[6:10])[2, 2])
        allocation = allocate(np.r_[thrust, torque], model=p)
        states.append(x.copy())
        errors.append(np.linalg.norm(error(x[6:10], [1, 0, 0, 0])))
        force_residual.append(allocation["residual"][0])
        moment_residual.append(np.linalg.norm(allocation["residual"][1:]))
        x = step(x, allocation["thrusts_n"], dt, model=p)
    a = {
        "time_s": t,
        "state": np.array(states),
        "attitude_error_rad": errors,
        "force_residual_n": force_residual,
        "moment_residual_nm": moment_residual,
    }
    return make(
        "spatial-attitude-recovery",
        {
            "model": asdict(p),
            "dt_s": dt,
            "initial_rotation_vector_rad": [0.25, -0.15, 0.05],
            "feedback": "ideal state benchmark",
        },
        {
            "final_attitude_error_rad": float(errors[-1]),
            "maximum_force_residual_n": float(max(abs(np.array(force_residual)))),
        },
        a,
        [
            curve(
                "Shortest attitude error",
                "time_s",
                [("attitude_error_rad", "Quaternion rotation error")],
                "Time (s)",
                "Angle (rad)",
            ),
            curve(
                "Rotor moment allocation residual",
                "time_s",
                [("moment_residual_nm", "Roll pitch yaw vector norm")],
                "Time (s)",
                "Moment (Nm)",
            ),
        ],
        assumptions=[
            "Separate spatial ideal-state attitude bench; no arm coupling or position-regulation claim."
        ],
    )
