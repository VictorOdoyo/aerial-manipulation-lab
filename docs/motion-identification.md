# Motion, identification and friction experiments
Rest-to-rest quintics return position, velocity, acceleration and jerk. Their exact peak factors are 1.875 for velocity, 10*sqrt(3)/3 for acceleration and 60 for jerk, each multiplied by displacement and the appropriate inverse duration power. The duration selector synchronizes coordinates under all three bounds. Piecewise rest quintics are C2; jerk may jump at waypoints. Analytical squared-jerk cost is checked against quadrature.

Continuous segment-to-circle clearance catches collisions missed by endpoint-only checks. It treats only a point end effector, not arm swept volume or aircraft collision. [Modern Robotics trajectory generation](https://modernrobotics.northwestern.edu/chapters/chapter9/) supplies path/time-scaling context.

Payload inverse dynamics is affine in unknown tip mass. A weighted nonnegative fit uses noisy joint-reaction observations, conditional on supplied configuration, velocity and acceleration measurements. Damping identification requires excitation in every fitted channel. Joint-only wrench reconstruction rejects singular geometry and reports conditional covariance, retaining its divergence near a straight arm.

Unilateral force projection enforces a Coulomb cone but may change normal force. The separate rigid inelastic impact model selects a consistent sticking/sliding impulse, enforces zero post-impact normal velocity, and rejects unsupported energy-generating branches. It matches the original frictionless normal impulse at zero friction.

These are synthetic standalone mechanics/design experiments, not a full spatial aircraft-arm model, online hardware identification, a global motion planner or a flight-certified contact controller.
