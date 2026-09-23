"""Thin-walled cylindrical artery under the membrane assumption (Section 3.1.2, Eqs. 21-22).

Given the growth tensor Fg = diag(lam_gr, lam_gt, lam_gz), the luminal pressure p and the
prescribed total axial stretch lam_z, the only unknown is the loaded mid-wall radius r_m
(equivalently the total circumferential stretch lam_t = r_m / R_m(0)), obtained from the
Laplace equation  sigma_tt = p r_i / h.
"""
from dataclasses import dataclass, field
import numpy as np

from ..units import wall_shear_stress, mmhg_to_kpa
from .fourfiber import FourFiberParams, four_fiber_membrane_stress
from .solvers import bisect_vec


@dataclass
class ThinWallArtery:
    D_o: float = 0.608            # unloaded outer diameter at day 0 [mm]
    H: float = 0.099              # unloaded wall thickness at day 0 [mm]
    lam_z: float = 1.81           # prescribed *total* axial stretch
    material: FourFiberParams = field(default_factory=FourFiberParams)
    Q_ml_min: float = 2.8         # mean flow rate [ml/min]
    mu_cP: float = 4.5            # blood viscosity [cP]
    bracket: tuple = (0.3, 4.0)   # bracket for lam_theta in the bisection

    @property
    def R_m(self):
        return 0.5 * (self.D_o - self.H)

    def solve(self, lam_g, p_kPa):
        """Solve equilibrium for a batch of growth states.

        Parameters
        ----------
        lam_g : (n, 3) array of growth stretches (r, theta, z)   [or (3,)]
        p_kPa : scalar or (n,) luminal pressure [kPa]

        Returns a dict of (n,) arrays: lam_theta, lam_r, r_m, r_i, h, sig_tt, sig_zz, tau_w,
        lam_er, lam_et, lam_ez, Jg.
        """
        lam_g = np.atleast_2d(np.asarray(lam_g, float))
        p = np.broadcast_to(np.asarray(p_kPa, float), (lam_g.shape[0],))
        lgr, lgt, lgz = lam_g[:, 0], lam_g[:, 1], lam_g[:, 2]
        Jg = lgr * lgt * lgz
        lam_ez = self.lam_z / lgz
        H, R_m, lz, mat = self.H, self.R_m, self.lam_z, self.material

        def geometry(x):
            lam_r = Jg / (x * lz)          # elastic incompressibility: lam_r lam_t lam_z = det(Fg)
            h = H * lam_r
            r_m = R_m * x
            r_i = r_m - 0.5 * h
            return lam_r, h, r_m, r_i

        def resid(x):
            lam_r, h, r_m, r_i = geometry(x)
            s_tt, _ = four_fiber_membrane_stress(lam_r / lgr, x / lgt, lam_ez, Jg, mat)
            return s_tt - p * r_i / h

        lo = np.full(lam_g.shape[0], self.bracket[0])
        hi = np.full(lam_g.shape[0], self.bracket[1])
        x = bisect_vec(resid, lo, hi, n_iter=45)
        lam_r, h, r_m, r_i = geometry(x)
        lam_er, lam_et = lam_r / lgr, x / lgt
        s_tt, s_zz = four_fiber_membrane_stress(lam_er, lam_et, lam_ez, Jg, mat)
        tau_w = wall_shear_stress(self.Q_ml_min, self.mu_cP, r_i)
        return dict(lam_theta=x, lam_r=lam_r, r_m=r_m, r_i=r_i, h=h, sig_tt=s_tt, sig_zz=s_zz,
                    tau_w=tau_w, lam_er=lam_er, lam_et=lam_et, lam_ez=lam_ez, Jg=Jg, p=p)

    def stress_state(self, lam_g, p_kPa):
        """Return the MDP stress state s = [sigma_tt, sigma_zz, tau_w] as an (n, 3) array."""
        out = self.solve(lam_g, p_kPa)
        return np.stack([out["sig_tt"], out["sig_zz"], out["tau_w"]], axis=1)
