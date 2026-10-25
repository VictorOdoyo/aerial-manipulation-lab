# Full spatial benches

The separate spatial quadrotor has 13 states: world position, world velocity, scalar-first body-to-world quaternion, and body angular rate. Euler rigid-body angular acceleration retains the gyroscopic cross product.

Quaternion composition, normalization, exponential integration and shortest rotation error have independent checks. The spatial RK4 integrator normalizes attitude after each accepted step. Torque-free precession preserves inertial angular momentum and mechanical energy within the reported numerical error.

Four-rotor plus-configuration allocation retains thrust, roll, pitch and yaw residuals when a rotor is unavailable. The attitude-recovery case is an ideal-state control benchmark; it is not an estimated-state position flight or a spatial arm manipulation claim.

The main aircraft-arm dynamics are coupled in the planar five-DoF model. Full three-dimensional arm coupling is outside this release. This separation is visible in study names, parameters and assumptions.
