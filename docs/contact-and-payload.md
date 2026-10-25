# Contact and payload events

The main arm carries a rigid point payload. Pickup and release are separate event benches with explicit mass changes. They are not silently injected into every flight replay.

Pickup solves generalized momentum matching using the old and new coupled mass matrices and the object's incoming velocity. Inelastic capture dissipates relative kinetic energy. Release keeps aircraft generalized velocity, gives the released object the current tip velocity and audits combined energy/momentum.

A unilateral plane contact API supplies a nonnegative spring-damper normal force. A separate frictionless impulse solve enforces restitution through the full inverse inertia. Joint-stop impulses likewise retain aircraft reaction rather than zeroing a joint rate in isolation.

The published capture/release, restitution and joint-stop studies verify these identities. There is no grasp planner, friction cone, compliant gripper identification or hardware contact-safety qualification.
