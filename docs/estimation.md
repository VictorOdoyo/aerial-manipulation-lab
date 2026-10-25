# Sensors and estimation

The accelerometer measures body-frame specific force, including gravity convention and declared bias/noise. It reads zero in free fall. The gyro includes bias and noise. Named random streams separate IMU, position, attitude reference and rotor tachometer draws.

Position measurements arrive every 0.1 s. An external attitude-reference measurement arrives every 0.02 s. These represent a generic laboratory reference sensor, not an undocumented GPS attitude capability. Encoders quantize relative joint angle, and filtered wrapped differences estimate joint rates.

An angle/gyro-bias filter predicts with measured rate and corrects wrapped attitude innovations. The navigation filter estimates planar position, velocity and world acceleration bias. Position corrections use normalized innovation gates and Joseph covariance form.

Sensor samples follow plant advancement; the corrected estimate is used for the next control interval. Truth is used only to generate simulated sensors and score estimates. Unavailable position intervals produce prediction without a zero-valued substitute. Invalid clipped accelerometer input uses the last acceleration with inflated uncertainty.

World-bias random walk approximates rotating body-frame bias. Noise and covariance parameters are illustrative rather than hardware calibrated. The observability bench shows why two position observations cannot determine all six navigation states.
