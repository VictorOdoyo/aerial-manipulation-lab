import numpy as np

from .core import finite


def metric(matrix):
    mass = finite(matrix, real=True, ndim=2)
    if mass.shape[0] != mass.shape[1] or not np.allclose(mass, mass.T) or min(np.linalg.eigvalsh(mass)) <= 0:
        raise ValueError("symmetric positive kinetic metric required")
    return mass


def tip(q, model=None):
    from .planar import kinematics

    return kinematics(q, model)["tip"]
