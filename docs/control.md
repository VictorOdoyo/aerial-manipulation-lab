# Coordinated control

Base position/velocity feedback requests acceleration. Relative-arm references include position, velocity and acceleration. Coupled inverse dynamics calculate aircraft torque and arm effort, with a bounded pitch reference.

One thrust direction cannot realize arbitrary independent x/z acceleration. The controller first projects the base request onto its physically available thrust direction, then recomputes attitude and arm efforts from the feasible acceleration. Hover trim retains a residual when a tilted configuration cannot balance gravity and external forces.

Rotor torque has a finite response time. Main arm compensation uses noisy rotor tachometer measurements and partitions the coupled mass matrix to predict aircraft acceleration from delivered rotor wrench. It does not assume requested torque appears instantly. Independent dynamics tests verify achieved arm acceleration under that measured wrench.

Allocation solves bounded weighted least squares. Torque residual divided by lever arm has force units; reported residual norms use this force-equivalent metric. Joint torque and slew limits remain active. Position-integral anti-windup uses feasible acceleration and allocated thrust.

The reaction-ignorant controller is a retained baseline. Ideal-state feedback is explicitly labeled a benchmark. Main default feedback is sensor derived. Kinematic task sharing is a separate bench and does not certify underactuated flight feasibility.
