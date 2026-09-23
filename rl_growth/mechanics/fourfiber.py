"""Four-fiber family constitutive model (Eq. 19) with mass-dependent remodelling (Eq. 20).

Used for the thin-walled murine infrarenal abdominal aorta (IAA) of Example 1.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class FourFiberParams:
    """Material parameters at day 0 (Table 1, Bersi et al. 2017) and remodelling constants.

    Fiber families: i=1 axial, i=2 circumferential, i=3,4 diagonal at +-alpha from the axial axis.
    """
    c: float = 10.646                                        # kPa
    c1: Tuple[float, float, float, float] = (7.838, 8.809, 0.217, 0.217)     # kPa
    c2: Tuple[float, float, float, float] = (0.130, 0.235, 0.865, 0.865)     # -
    alpha_deg: float = 35.372
    k1: Tuple[float, float, float, float] = (0.624, 0.486, 0.704, 0.704)
    k2: Tuple[float, float, float, float] = (-4.557, -4.513, 0.396, 0.396)
    remodeling: bool = True        # apply Eq. (20)
    tension_only: bool = False     # if True fibers carry load only when I4 > 1

    @property
    def fiber_angles_rad(self):
        a = np.deg2rad(self.alpha_deg)
        return (0.0, 0.5 * np.pi, a, -a)


def remodeled_parameters(params: FourFiberParams, Jg):
    """Eq. (20): c^i(t) = c^i(0) * max{ k^i [det(Fg) - 1] + 1, 0 }."""
    Jg = np.asarray(Jg, dtype=float)
    c1 = []
    c2 = []
    for i in range(4):
        if params.remodeling:
            c1.append(params.c1[i] * np.maximum(params.k1[i] * (Jg - 1.0) + 1.0, 0.0))
            c2.append(params.c2[i] * np.maximum(params.k2[i] * (Jg - 1.0) + 1.0, 0.0))
        else:
            c1.append(params.c1[i] * np.ones_like(Jg))
            c2.append(params.c2[i] * np.ones_like(Jg))
    return c1, c2


def four_fiber_membrane_stress(lam_er, lam_et, lam_ez, Jg, params: FourFiberParams):
    """Cauchy stresses (sigma_tt, sigma_zz) [kPa] of an incompressible membrane (sigma_rr = 0).

    Parameters are broadcastable arrays of *elastic* stretches (radial, circumferential, axial)
    and det(Fg) for the remodelling law.
    """
    lam_er = np.asarray(lam_er, float)
    lam_et = np.asarray(lam_et, float)
    lam_ez = np.asarray(lam_ez, float)
    dWdI1 = 0.5 * params.c
    # plane stress: Lagrange multiplier p = 2 lam_er^2 dW/dI1
    s_tt = 2.0 * (lam_et ** 2 - lam_er ** 2) * dWdI1
    s_zz = 2.0 * (lam_ez ** 2 - lam_er ** 2) * dWdI1
    c1s, c2s = remodeled_parameters(params, Jg)
    for i, ang in enumerate(params.fiber_angles_rad):
        s2, cs2 = np.sin(ang) ** 2, np.cos(ang) ** 2
        I4 = lam_et ** 2 * s2 + lam_ez ** 2 * cs2
        E = I4 - 1.0
        fib = c1s[i] * E * np.exp(c2s[i] * E ** 2)       # = 2 dW/dI4
        if params.tension_only:
            fib = np.where(E > 0.0, fib, 0.0)
        s_tt = s_tt + fib * lam_et ** 2 * s2
        s_zz = s_zz + fib * lam_ez ** 2 * cs2
    return s_tt, s_zz
