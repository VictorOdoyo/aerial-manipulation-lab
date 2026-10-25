# Experiment catalog

The release contains 13 coupled-flight cases: reaction-aware arm motion, sensor-feedback hover, reaction-ignorant baseline, coordinated translation, gust disturbance, position outage, position outlier, payload mismatch, known heavy payload, partial actuator loss, complete actuator loss, low pack headroom and an explicitly ideal-state benchmark.

Fourteen independent benches cover energy convergence, internal momentum, hover trim, rotor lag, capture/release, contact restitution, arm manipulability, joint stops, kinematic task sharing, spatial free rotation, spatial attitude recovery, spatial failure allocation, inertial outage and navigation observability.

Use the computed gallery for measured numerical outputs and its per-case assumptions. Tracking RMSE, final joint error, conservation residuals, allocation residuals and termination are different diagnostics. A low algebraic residual alone does not establish physical accuracy.

Actuator loss can make a desired flight unattainable. The gallery retains actual truncated trajectories and residuals rather than changing the demand or assigning a successful label.
