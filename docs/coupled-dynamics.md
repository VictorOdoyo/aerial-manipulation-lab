# Coupled floating-base dynamics

The main model is planar and has five generalized coordinates: x, z, aircraft pitch, relative joint 1, relative joint 2. z is upward. Link directions are measured from downward vertical; their absolute angles are pitch+q1 and pitch+q1+q2. Positive aircraft pitch rotates body thrust toward negative x.

The mass matrix is assembled from each body center-of-mass Jacobian and rotational kinetic energy. Links are uniform rods, and payload is a point mass at the tip. Aircraft translation and rotation, both link inertias, and cross-coupling terms are retained.

Dynamics satisfy M(q) qdd + C(q,qd) qd + G(q) = Q. Gravity is the gradient of potential energy. Christoffel terms use complex-step mass derivatives; cyclic translation and shared pitch/joint-1 dependence reduce redundant evaluations. Independent centered differences check the derivatives, and Mdot-2C is skew symmetric.

The coupled solver audits kinetic+potential energy against integrated generalized-force work. Internal joint torques react on the aircraft while conserving external momentum. A free-fall check verifies that the whole articulated system accelerates at gravity without fictitious joint motion.

This is a numerical model, not an identified aircraft. Formulation context: [MIT underactuated multibody dynamics](https://underactuated-r1.csail.mit.edu/multibody.html), [MIT quadrotors and underactuation](https://underactuated.mit.edu/acrobot.html). Implementation and tests are independently authored.
