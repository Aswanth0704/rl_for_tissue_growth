"""Unit helpers.  Internal units: length mm, stress kPa, wall shear stress dyn/cm^2, time days."""
import numpy as np

MMHG_TO_KPA = 0.1333223684


def mmhg_to_kpa(p_mmhg):
    return p_mmhg * MMHG_TO_KPA


def wall_shear_stress(Q_ml_min, mu_cP, r_i_mm):
    """Poiseuille wall shear stress (Eq. 4), returned in dyn/cm^2.

    Parameters
    ----------
    Q_ml_min : volume flow rate [ml/min]
    mu_cP    : blood viscosity [cP]
    r_i_mm   : loaded inner radius [mm]
    """
    Q = Q_ml_min / 60.0        # cm^3/s
    mu = mu_cP / 100.0         # poise = dyn s / cm^2
    r = np.asarray(r_i_mm) / 10.0   # cm
    return 4.0 * mu * Q / (np.pi * r ** 3)
