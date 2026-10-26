"""Validated reproducible run requests and lazy study dispatch."""

import json
from dataclasses import dataclass
from pathlib import Path

CASES = {
    "reaction-aware-arm-motion": ["studies_flight", "nominal", "Causal reaction-aware manipulation"],
    "sensor-feedback-hover": ["studies_flight", "hover", "Sensor-derived hovering state"],
    "reaction-ignorant-baseline": ["studies_flight", "baseline", "Ignored arm reaction challenge"],
    "coordinated-translation": ["studies_flight", "coordinated", "Base and arm motion together"],
    "wind-gust-rejection": ["studies_flight", "wind", "Aerodynamic gust disturbance"],
    "position-sensor-outage": ["studies_flight", "outage", "Inertial coasting without position"],
    "position-outlier-gating": ["studies_flight", "outlier", "Rejected position outlier"],
    "payload-model-mismatch": ["studies_flight", "payload_mismatch", "Unmodeled payload inertia"],
    "known-heavy-payload": ["studies_flight", "payload_known", "Correctly modeled heavy payload"],
    "partial-actuator-loss": ["studies_flight", "partial_loss", "Partial rotor-pair power loss"],
    "complete-actuator-loss": ["studies_flight", "complete_loss", "Complete rotor-pair power loss"],
    "low-pack-headroom": ["studies_flight", "low_pack", "Voltage sag and rotor headroom"],
    "ideal-state-feedback-benchmark": ["studies_flight", "ideal", "Explicit ideal-state benchmark"],
    "coupled-energy-convergence": [
        "studies_mechanics",
        "energy_convergence",
        "Independent conservation under refinement",
    ],
    "internal-reaction-momentum": [
        "studies_mechanics",
        "internal_reaction",
        "Base reaction with conserved momentum",
    ],
    "hover-trim-workspace": ["studies_mechanics", "trim_workspace", "Payload and actuator-limited hover"],
    "rotor-lag-response": ["studies_mechanics", "rotor_lag", "Exact mean rotor thrust"],
    "payload-capture-release": [
        "studies_mechanics",
        "payload_sequence",
        "Capture and release momentum audit",
    ],
    "contact-impact-restitution": ["studies_mechanics", "contact_restitution", "Coupled contact impulses"],
    "arm-manipulability": ["studies_mechanics", "manipulability_curve", "Independent arm workspace Jacobian"],
    "joint-stop-base-reaction": ["studies_mechanics", "joint_impact", "Joint stop reacts on aircraft"],
    "kinematic-task-sharing": ["studies_mechanics", "task_sharing", "Weighted kinematic redundancy"],
    "spatial-free-rotation": ["studies_spatial", "free_rotation", "3D rigid-body invariants"],
    "spatial-attitude-recovery": [
        "studies_spatial",
        "attitude_recovery",
        "Bounded four-rotor attitude bench",
    ],
    "quadrotor-failure-allocation": ["studies_spatial", "quad_loss", "Unavailable spatial wrench authority"],
    "inertial-position-outage": ["studies_sensing", "outage", "Sensor-only inertial observer"],
    "navigation-observability": ["studies_sensing", "observability", "Position and bias observability"],
    "task-inertia-payload": [
        "studies_taskspace",
        "inertia",
        "Measure payload-dependent Cartesian inertia using full floating-base mechanics",
    ],
    "dynamic-task-velocity": [
        "studies_taskspace",
        "velocity",
        "Compare kinetic-energy-optimal task motion with Euclidean pseudoinverse allocation",
    ],
    "nullspace-joint-barrier": [
        "studies_taskspace",
        "nullspace",
        "Reduce a joint-limit barrier while preserving instantaneous Cartesian task velocity",
    ],
    "bilateral-contact-reaction": [
        "studies_taskspace",
        "constraint",
        "Solve coupled bilateral tip constraints and audit the reaction force",
    ],
    "impedance-virtual-work": [
        "studies_taskspace",
        "work",
        "Audit Cartesian impedance and generalized reaction power over random coupled motions",
    ],
    "task-force-authority": [
        "studies_taskspace",
        "authority",
        "Scale requested task forces against declared generalized-force bounds",
    ],
    "bounded-quintic-motion": [
        "studies_taskspace",
        "retiming",
        "Retime synchronized joint motion to satisfy velocity acceleration and jerk limits",
    ],
    "rest-to-rest-waypoints": [
        "studies_taskspace",
        "waypoints",
        "Verify continuous position velocity and acceleration through synchronized rest waypoints",
    ],
    "continuous-path-clearance": [
        "studies_taskspace",
        "clearance",
        "Detect obstacle intersections between samples and verify a declared waypoint detour",
    ],
    "quintic-jerk-tradeoff": [
        "studies_taskspace",
        "jerk",
        "Compare analytical and sampled integrated jerk cost as motion duration changes",
    ],
    "payload-mass-identification": [
        "studies_taskspace",
        "payload",
        "Identify unknown payload mass from noisy arm-joint inverse-dynamics reactions",
    ],
    "viscous-damping-identification": [
        "studies_taskspace",
        "damping",
        "Recover passive damping coefficients from excited velocity and residual-force observations",
    ],
    "contact-wrench-conditioning": [
        "studies_taskspace",
        "wrench",
        "Retain force-observation singularity near a straight arm with joint-only torque sensing",
    ],
    "unilateral-force-projection": [
        "studies_taskspace",
        "cone",
        "Project infeasible contact forces into the unilateral Coulomb cone",
    ],
    "coupled-friction-impact": [
        "studies_taskspace",
        "impact",
        "Resolve sticking and sliding impacts through the floating-base kinetic metric",
    ],
}


@dataclass(frozen=True)
class ExperimentConfig:
    case: str = "reaction-aware-arm-motion"
    resolution_scale: int = 1
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.case, str) or self.case not in CASES:
            raise ValueError("unknown experiment case")
        if type(self.resolution_scale) is not int or not 1 <= self.resolution_scale <= 3:
            raise ValueError("resolution_scale must be an integer from 1 to 3")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")

    def to_dict(self):
        return {"case": self.case, "resolution_scale": self.resolution_scale, "seed": self.seed}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"case", "resolution_scale", "seed"}:
            raise ValueError("request must be an object containing only case, resolution_scale and seed")
        return cls(**value)


def load_config(path):
    return ExperimentConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def run(config, progress=None):
    from importlib import import_module

    from .reporting import provenance, stamp
    from .results import validate

    if not isinstance(config, ExperimentConfig):
        raise ValueError("validated ExperimentConfig required")
    configuration = config.to_dict()
    source = provenance(configuration)
    module, function, _ = CASES[config.case]
    if progress is not None:
        progress("Solving " + config.case)
    result = getattr(import_module("aeriallab." + module), function)(
        scale=config.resolution_scale, seed=config.seed
    )
    validate(result)
    if result["name"] != config.case:
        raise ValueError("study returned a different case name")
    return stamp(result, configuration, source)
