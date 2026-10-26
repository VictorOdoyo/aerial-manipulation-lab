def payload_regressor(q, velocity, acceleration, model=None):
    from dataclasses import replace

    from .planar import Model, configuration, coriolis_matrix, gravity, mass_matrix

    p = Model() if model is None else model
    x = configuration(q)
    v = configuration(velocity)
    a = configuration(acceleration)

    def inverse(mass):
        candidate = replace(p, payload_kg=mass)
        return mass_matrix(x, candidate) @ a + coriolis_matrix(x, v, candidate) @ v + gravity(x, candidate)

    baseline = inverse(0)
    return {"baseline": baseline, "column": inverse(1) - baseline}
