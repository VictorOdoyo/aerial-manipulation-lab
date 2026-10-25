# Architecture

| Layer | Modules | Responsibility |
|---|---|---|
| Coupled mechanics | planar, integration, arm | floating-base inertia, dynamics, task kinematics and work audit |
| Interaction | payload, contact, environment | momentum events, impacts, wind and drag |
| Actuation | actuators, electrics | rotor/joint limits, speed lag, battery and power |
| Feedback | trajectory, control, sensors, estimation | causal references, measured-wrench compensation and observer |
| Spatial benches | rotation, quadrotor | separate full 3D quaternion flight and allocation |
| Evidence | simulation, studies_flight, studies_mechanics, studies_spatial, studies_sensing | reproducible physical cases |
| Artifacts | experiments, results, reporting, batch, cli | strict requests, source freezing, archives and replay |

Artifact infrastructure is adapted from this author's MIT-licensed ev-powertrain-lab, radar-perception-stack and adaptive-grid-protection work. Aerial dynamics, sensor feedback, payload events and studies are independently authored for this repository.
