"""Gasser-Ogden-Holzapfel (GOH) constitutive model (Eq. 23) for the bi-layer human ascending aorta."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GOHParams:
    c: float          # kPa
    c1: float         # kPa
    c2: float         # -
    kappa: float      # fibre dispersion, in (0, 1/3)
    alpha_deg: float  # mean fibre angle measured from the circumferential axis
    tension_only: bool = True   # fibres contribute only when E > 0 (standard GOH)


# Table 3 (Liu et al. 2019 / this paper)
MEDIA_GOH = GOHParams(c=46.0194, c1=127.0692, c2=4.4952, kappa=0.3201, alpha_deg=0.0008)
ADVENTITIA_GOH = GOHParams(c=16.5298, c1=71.2311, c2=1.6901, kappa=0.3013, alpha_deg=0.0010)


def goh_extra_stress(lam_er, lam_et, lam_ez, params: GOHParams):
    """'Extra' Cauchy stresses (without the Lagrange multiplier) in principal directions (r, t, z).

    sigma = -p I + sigma_hat ;   sigma_hat_j = 2 lam_j^2 dW/dI1 + fibre terms.
    Returns (s_rr_hat, s_tt_hat, s_zz_hat) in kPa.
    """
    lam_er = np.asarray(lam_er, float)
    lam_et = np.asarray(lam_et, float)
    lam_ez = np.asarray(lam_ez, float)
    a = np.deg2rad(params.alpha_deg)
    ca2, sa2 = np.cos(a) ** 2, np.sin(a) ** 2
    I1 = lam_er ** 2 + lam_et ** 2 + lam_ez ** 2
    I4 = lam_et ** 2 * ca2 + lam_ez ** 2 * sa2          # identical for the +-alpha families
    k = params.kappa
    E = k * I1 + (1.0 - 3.0 * k) * I4 - 1.0
    g = params.c1 * E * np.exp(params.c2 * E ** 2)      # dW_fib/dE per family
    if params.tension_only:
        g = np.where(E > 0.0, g, 0.0)
    dWdI1 = 0.5 * params.c + 2.0 * k * g                # two families
    dWdI4 = (1.0 - 3.0 * k) * g                         # per family
    s_rr = 2.0 * lam_er ** 2 * dWdI1
    s_tt = 2.0 * lam_et ** 2 * dWdI1 + 2.0 * (2.0 * lam_et ** 2 * ca2 * dWdI4)
    s_zz = 2.0 * lam_ez ** 2 * dWdI1 + 2.0 * (2.0 * lam_ez ** 2 * sa2 * dWdI4)
    return s_rr, s_tt, s_zz
