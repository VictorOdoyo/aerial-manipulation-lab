"""Independent component examples using the installed aeriallab package."""

import numpy as np

from aeriallab.control import hover_trim
from aeriallab.payload import pickup
from aeriallab.planar import Model, kinematics, mass_matrix


def main():
    model = Model()
    q = np.array([0.0, 2.0, 0.0, 0.5, -0.6])
    print("Tip (m):", kinematics(q, model)["tip"][0])
    print("Mass matrix eigenvalues:", np.linalg.eigvalsh(mass_matrix(q, model)))
    print("Hover trim:", hover_trim(q, model))
    state = np.r_[q, [1.0, 0.0, 0.1, 0.2, -0.1]]
    captured = pickup(state, 0.2, model=model)
    print("Capture dissipation (J):", captured["dissipated_j"])


if __name__ == "__main__":
    main()
