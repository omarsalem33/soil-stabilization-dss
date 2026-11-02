"""Prediction helpers and the app's explicitly simplified footing estimate."""

import numpy as np

def predict_responses(models, row):
    return {target: float(model.predict(row)[0]) for target, model in models.items()}

def simplified_allowable_pressure(cohesion_kpa, phi_degrees, width_m, depth_m, unit_weight_kn_m3, factor_of_safety):
    """Return the exploratory general-shear estimate used by the UI; not for design."""
    phi = np.radians(max(0.0, min(45.0, phi_degrees)))
    if phi < 1e-6:
        nq, nc, ngamma = 1.0, 5.14, 0.0
    else:
        nq = np.exp(np.pi * np.tan(phi)) * np.tan(np.pi / 4 + phi / 2) ** 2
        nc = (nq - 1) / np.tan(phi)
        ngamma = 2 * (nq + 1) * np.tan(phi)
    ultimate = (cohesion_kpa * nc + unit_weight_kn_m3 * depth_m * nq
                + 0.5 * unit_weight_kn_m3 * width_m * ngamma)
    return ultimate / factor_of_safety
