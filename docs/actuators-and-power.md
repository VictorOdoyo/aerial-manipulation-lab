# Actuation and electrical power

The coupled planar aircraft uses two aggregate lateral rotor pairs. Their sum produces thrust; their difference times the 0.25 m lever arm produces pitch torque. The separate spatial quadrotor has four individual plus-configuration rotors.

Rotor speed follows an exact first-order lag. Thrust is proportional to squared speed, and interval-averaged thrust is integrated analytically. Voltage and health constrain target speed; an unpowered rotor can coast down from its previous speed. Joint efforts have amplitude and slew limits.

Electrical propulsion power combines ideal induced power, a declared profile term and efficiency. Joint motoring draws electrical power; braking energy is dissipated rather than credited as regeneration. Auxiliary loads are explicit.

The flight pack uses an illustrative six-series-cell OCV/SOC relation, constant resistance, a stable low-current quadratic constant-power solution and coulomb counting. Voltage affects rotor headroom. This is not a switched motor/inverter, cell chemistry or thermal pack model.

Mechanical work and electrical draw are different quantities. Propulsive induced losses, servo losses and auxiliary consumption prevent battery energy from equaling aircraft rigid-body kinetic energy. All resolved coefficients are included in study parameters.
