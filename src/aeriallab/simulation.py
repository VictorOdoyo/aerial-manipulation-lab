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
    outlier_time_s: float = 3.0
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
        positive(self.outlier_time_s, allow_zero=True)
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


def run(config):
    from .actuators import (
        allocate,
        joint_effort,
        mean_thrust,
        rotor_step,
        speed_ceiling,
        wrench,
        wrench_error_n,
    )
    from .control import PositionIntegral, arm_effort_for_wrench, command
    from .core import timeline
    from .electrics import battery_point, battery_step, propulsion_power, servo_power
    from .environment import gust
    from .estimation import Navigation
    from .integration import wrench_step
    from .planar import Model, kinematics
    from .sensors import EncoderVelocity, Suite, encoder, rotor_tachometer
    from .trajectory import reference

    if not isinstance(config, Config):
        raise ValueError("validated flight replay config required")
    plant = Model(payload_kg=config.payload_kg)
    controller_model = Model(payload_kg=config.assumed_payload_kg)
    time = timeline(config.duration_s, config.dt_s)
    n = len(time)
    state = np.r_[[0.0, 2.0, 0.0, 0.0, 0.0], np.zeros(5)]
    sensors = Suite(config.seed)
    nav = Navigation()
    enc_rate = EncoderVelocity()
    enc = encoder(state[3:5])["angle_rad"]
    joint_rate = enc_rate.update(enc, config.dt_s)
    rotor_speed = np.full(2, np.sqrt(controller_model.total_mass * plant.gravity_m_s2 / 2 / 2e-5))
    previous_joint = np.zeros(2)
    pack_soc = config.initial_soc
    bus = battery_point(pack_soc, 0)["bus_v"]
    gyro = 0.0
    integral = PositionIntegral(gain=0.3 if config.integral else 0)
    shapes = {
        "state": (10,),
        "estimated_state": (10,),
        "reference_position_m": (2,),
        "reference_arm_rad": (2,),
        "rotor_thrust_n": (2,),
        "joint_torque_nm": (2,),
        "tip_m": (2,),
        "navigation_covariance": (6, 6),
    }
    arrays = {key: np.zeros((n, *shape)) for key, shape in shapes.items()}
    for key in (
        "battery_soc",
        "bus_v",
        "battery_current_a",
        "electric_w",
        "mechanical_defect_j",
        "pitch_reference_rad",
        "allocation_residual_n",
        "position_nis",
        "wind_m_s",
    ):
        arrays[key] = np.zeros(n)
    for key in ("position_available", "position_accepted", "imu_valid"):
        arrays[key] = np.zeros(n, bool)
    arrays["state"][0] = state
    arrays["estimated_state"][0] = np.r_[nav.state[:2], 0.0, enc, nav.state[2:4], 0.0, joint_rate]
    arrays["battery_soc"][0] = pack_soc
    arrays["bus_v"][0] = bus
    arrays["tip_m"][0] = kinematics(state[:5], plant)["tip"][0]
    arrays["navigation_covariance"][0] = nav.covariance
    initial_reference = reference(0, config.moving_base, config.moving_arm)
    arrays["reference_position_m"][0] = initial_reference["base"]["position"]
    arrays["reference_arm_rad"][0] = initial_reference["arm"]["position"]
    next_position = 0.1
    next_attitude = 0.02
    outlier_used = False
    reason = "completed"
    end = n
    for k in range(1, n):
        dt = time[k] - time[k - 1]
        t = time[k - 1]
        ref = reference(t, config.moving_base, config.moving_arm)
        feedback = (
            np.r_[
                nav.state[:2],
                nav.attitude.state[0],
                enc,
                nav.state[2:4],
                gyro - nav.attitude.state[1],
                joint_rate,
            ]
            if config.feedback == "estimated"
            else state.copy()
        )
        requested = command(feedback, ref, controller_model, config.reaction_compensation, integral.value)
        health = config.health_after if config.actuator_fault and t >= config.fault_time_s else (1, 1)
        ceiling = speed_ceiling(bus, health=health)
        maximum = 2e-5 * ceiling**2
        allocation = allocate(requested["thrust_n"], requested["body_torque_nm"], maximum_n=maximum)
        joint_request = requested["joint_torques_nm"]
        if config.reaction_compensation:
            measured_wrench = wrench(2e-5 * rotor_tachometer(sensors, rotor_speed) ** 2)
            joint_request = arm_effort_for_wrench(
                feedback, requested["joint_acceleration_rad_s2"], measured_wrench, controller_model
            )["torque_nm"]
        joint = joint_effort(joint_request, previous_joint, dt)["torque_nm"]
        previous_joint = joint
        average_thrust = mean_thrust(rotor_speed, allocation["thrusts_n"], dt, maximum_rad_s=ceiling)
        rotors = rotor_step(rotor_speed, allocation["thrusts_n"], dt, maximum_rad_s=ceiling)
        rotor_speed = rotors["speed_rad_s"]
        actual_wrench = wrench(average_thrust)
        wind = float(gust(t, peak_m_s=config.wind_peak_m_s))
        old = state.copy()
        advanced = wrench_step(
            state, actual_wrench[0], actual_wrench[1], joint, dt, plant, wind_velocity=(wind, 0)
        )
        state = advanced["state"]
        electrical = (
            propulsion_power(average_thrust, rotor_speed)["electrical_w"]
            + servo_power(joint, (state[8:] + old[8:]) / 2)["electrical_w"]
            + 20
        )
        pack = battery_point(pack_soc, electrical)
        pack_soc = battery_step(pack_soc, pack["current_a"], dt)
        bus = pack["bus_v"]
        actual_acceleration = (state[5:7] - old[5:7]) / dt
        imu = sensors.imu(state[2], state[7], actual_acceleration, plant.gravity_m_s2)
        gyro = imu["gyro_rad_s"]
        position = None
        attitude_reference = None
        available = time[k] + 1e-10 >= next_position
        if available:
            next_position += 0.1
            in_outage = (
                config.position_outage is not None
                and config.position_outage[0] <= time[k] <= config.position_outage[1]
            )
            if not in_outage:
                position = sensors.position(state[:2])
                if config.position_outlier and time[k] >= config.outlier_time_s and not outlier_used:
                    position += np.array([3.0, 3.0])
                    outlier_used = True
        if time[k] + 1e-10 >= next_attitude:
            attitude_reference = sensors.attitude(state[2])
            next_attitude += 0.02
        result = nav.update(imu, dt, position, attitude_reference, plant.gravity_m_s2)
        enc = encoder(state[3:5])["angle_rad"]
        joint_rate = enc_rate.update(enc, dt)
        direction = np.array([-np.sin(feedback[2]), np.cos(feedback[2])])
        requested_acc = requested["base_acceleration_m_s2"]
        achieved_acc = (
            requested_acc
            + (allocation["achieved"][0] - requested["thrust_n"]) * direction / controller_model.total_mass
        )
        integral.update(ref["base"]["position"] - feedback[:2], dt, requested_acc, achieved_acc)
        arrays["state"][k] = state
        arrays["estimated_state"][k] = np.r_[
            nav.state[:2],
            nav.attitude.state[0],
            enc,
            nav.state[2:4],
            gyro - nav.attitude.state[1],
            joint_rate,
        ]
        display_ref = reference(time[k], config.moving_base, config.moving_arm)
        arrays["reference_position_m"][k] = display_ref["base"]["position"]
        arrays["reference_arm_rad"][k] = display_ref["arm"]["position"]
        arrays["rotor_thrust_n"][k] = average_thrust
        arrays["joint_torque_nm"][k] = joint
        arrays["tip_m"][k] = kinematics(state[:5], plant)["tip"][0]
        arrays["navigation_covariance"][k] = nav.covariance
        values = {
            "battery_soc": pack_soc,
            "bus_v": bus,
            "battery_current_a": pack["current_a"],
            "electric_w": electrical,
            "mechanical_defect_j": advanced["mechanical_defect_j"],
            "pitch_reference_rad": requested["pitch_reference_rad"],
            "allocation_residual_n": wrench_error_n(allocation["residual"]),
            "position_nis": result["position_nis"],
            "wind_m_s": wind,
            "position_available": position is not None,
            "position_accepted": result["position_accepted"],
            "imu_valid": result["imu_valid"],
        }
        for key, value in values.items():
            arrays[key][k] = value
        if state[1] < 0.2 or abs(state[2]) > 1.4 or pack_soc < 0.05:
            reason = "flight-envelope-exceeded"
            end = k + 1
            break
    arrays = {key: value[:end] for key, value in arrays.items()}
    arrays["time_s"] = time[:end]
    return {
        "arrays": arrays,
        "termination": reason,
        "configuration": config.to_dict(),
        "plant": asdict(plant),
        "controller_model": asdict(controller_model),
    }
