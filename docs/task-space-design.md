# Coupled task-space mechanics
The new design benches use the existing planar five-coordinate floating-base mass matrix. They remain separate from the sampled aircraft/arm flight controller.

For tip Jacobian J and kinetic metric M, Cartesian inertia is Lambda=(J M^-1 J^T)^-1. The dynamically consistent inverse is M^-1 J^T Lambda and its null projector preserves task velocity. A kinetic-metric joint-barrier gradient provides an instantaneous descent direction without tip leakage. Task/generalized power equality is tested independently.

The constrained-acceleration bench solves a bilateral KKT system including convective Jdot*v. Bilateral contact can pull or push; it is not a unilateral ground model. Cartesian impedance maps forces with J^T and checks virtual work. Generalized-force box limits are an idealized design bound; they do not provide missing aircraft actuation or replace rotor allocation.

Whole-system COM includes link and payload mass. Payload changes modify Cartesian inertia. Minimum-energy velocity allocation is an instantaneous kinematic/mechanical comparison and does not certify a dynamically feasible flight.

See [Stanford's operational-space framework](https://khatib.stanford.edu/projects/controlandsimulation.html), [motion and identification](motion-identification.md), and the [computed gallery](results.md).
