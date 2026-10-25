# Verification scope

Independent checks include mass-matrix symmetry/positivity, analytic versus differentiated Jacobians, potential gradients, Coriolis skew identity, free fall, instantaneous power, conservative integration convergence, momentum, contact restitution and payload-event accounting.

Control checks cover infeasible hover trim, underactuated acceleration projection, actual-wrench arm compensation, rotor lag, bounded allocation, encoder wrap, noisy sensor streams, covariance positivity, outlier rejection and achieved coordinated flight.

Challenge studies retain model mismatch, actuator loss, observation outages and envelope termination. A terminated trajectory ends at its actual last state; missing time is not filled with successful-looking zeros. Default sensor-feedback cases and ideal-state benchmarks are distinguished explicitly.

These are synthetic numerical experiments with illustrative coefficients. No real aircraft calibration, flight certification, autonomous safety guarantee, measured battery validation or spatial coupled-arm implementation is claimed.

CI checks supported Python versions on Linux/Windows, formatting, all published evidence and an installed-wheel scientific report outside the checkout.
