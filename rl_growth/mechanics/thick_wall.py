"""Bi-layer thick-walled cylinder with heterogeneous growth (Section 3.2.2, Eqs. 24-27).

Each layer k (media, adventitia) carries n_gauss integration points (Table 4) at which an
independent growth tensor Fg = diag(lam_gr, lam_gt, lam_gz) lives.  The kinematics of
Taber & Humphrey (2001) reduce equilibrium to a single scalar unknown (loaded inner radius)
for the pressurised, axially stretched vessel.  The same machinery with the sector angle and
the axial stretch as unknowns gives the unloaded (residually stressed) and the cut-open
(opening angle) configurations.

Growth arrays have shape (batch, n_layers, n_gauss, 3).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.optimize import root

from ..units import wall_shear_stress, mmhg_to_kpa
from .goh import GOHParams, MEDIA_GOH, ADVENTITIA_GOH, goh_extra_stress
from .solvers import bisect_vec

GL5_NODES = np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                      0.5384693101056831, 0.9061798459386640])
GL5_WEIGHTS = np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                        0.4786286704993665, 0.2369268850561891])


def _cumint_matrix(nodes):
    """Matrix C with (C f)_i = int_{-1}^{x_i} L[f](x) dx, L = Lagrange interpolant on ``nodes``."""
    n = len(nodes)
    C = np.zeros((n, n))
    for j in range(n):
        coef = P.polyfromroots(np.delete(nodes, j))
        coef = coef / P.polyval(nodes[j], coef)
        integ = P.polyint(coef)
        C[:, j] = P.polyval(nodes, integ) - P.polyval(-1.0, integ)
    return C


@dataclass
class Layer:
    name: str
    R_i: float
    R_o: float
    material: GOHParams


@dataclass
class BilayerAorta:
    layers: List[Layer] = field(default_factory=lambda: [
        Layer("media", 5.61, 6.43, MEDIA_GOH),
        Layer("adventitia", 6.43, 7.05, ADVENTITIA_GOH)])
    p_mmHg: float = 96.0          # weighted mean blood pressure
    lam_z: float = 1.2            # in vivo axial stretch (total, before growth)
    Q_ml_min: float = 856.5
    mu_cP: float = 4.5

    def __post_init__(self):
        L = len(self.layers)
        self.n_layers = L
        self.n_gauss = 5
        self.R = np.zeros((L, 5))
        self.wts = np.zeros((L, 5))
        self.C = np.zeros((L, 5, 5))
        Cref = _cumint_matrix(GL5_NODES)
        for k, lay in enumerate(self.layers):
            half = 0.5 * (lay.R_o - lay.R_i)
            self.R[k] = lay.R_i + (GL5_NODES + 1.0) * half
            self.wts[k] = GL5_WEIGHTS * half
            self.C[k] = Cref * half
        self.p_kPa = mmhg_to_kpa(self.p_mmHg)

    # ------------------------------------------------------------------ helpers
    def identity_growth(self, batch=1):
        return np.ones((batch, self.n_layers, self.n_gauss, 3))

    def layer_mean(self, values, layer_ids=None):
        """Transmural mean over the reference thickness: int f dR / (R_o - R_i)  -> (batch, L)."""
        ids = list(range(self.n_layers)) if layer_ids is None else list(layer_ids)
        wts = self.wts[ids]
        thick = np.array([self.layers[k].R_o - self.layers[k].R_i for k in ids])
        return (values * wts).sum(-1) / thick

    # ------------------------------------------------------------------ core
    def evaluate(self, lam_g, r_i, Lam, Phi=2.0 * np.pi, p_kPa=0.0, layer_ids=None):
        """Evaluate kinematics, stresses and global residuals for a batch of configurations.

        Parameters
        ----------
        lam_g : (B, L', 5, 3) growth stretches for the layers in ``layer_ids``
        r_i   : (B,) deformed inner radius of the innermost included layer
        Lam   : (B,) total axial stretch
        Phi   : scalar or (B,) sector angle of the deformed configuration (2 pi = closed)
        p_kPa : scalar or (B,) luminal pressure
        layer_ids : layers included (default: all).  Use a single layer for opening angles.
        """
        ids = list(range(self.n_layers)) if layer_ids is None else list(layer_ids)
        lam_g = np.asarray(lam_g, float)
        B = lam_g.shape[0]
        r_i = np.broadcast_to(np.asarray(r_i, float), (B,))
        Lam = np.broadcast_to(np.asarray(Lam, float), (B,))
        Phi = np.broadcast_to(np.asarray(Phi, float), (B,))
        p = np.broadcast_to(np.asarray(p_kPa, float), (B,))
        R, wts, C = self.R[ids], self.wts[ids], self.C[ids]
        nL = len(ids)

        Jg = lam_g[..., 0] * lam_g[..., 1] * lam_g[..., 2]          # (B, L, 5)
        # ---- radial mapping, Eq. (26) generalised to a sector of angle Phi:
        #      rho^2 - rho_i^2 = 4 pi / (Phi Lam) * int det(Fg) R dR
        f = Jg * R[None]
        fac = 4.0 * np.pi / (Phi * Lam)                                 # (B,)
        cum = np.einsum("lij,blj->bli", C, f)
        tot = (f * wts[None]).sum(-1)                                   # (B, L)
        r_in = np.empty((B, nL))
        r_in[:, 0] = r_i
        for k in range(1, nL):
            r_in[:, k] = np.sqrt(r_in[:, k - 1] ** 2 + fac * tot[:, k - 1])
        r = np.sqrt(r_in[..., None] ** 2 + fac[:, None, None] * cum)   # (B, L, 5)
        r_out = np.sqrt(r_in[:, -1] ** 2 + fac * tot[:, -1])
        lam_t = (Phi[:, None, None] / (2.0 * np.pi)) * r / R[None]
        lam_zz = Lam[:, None, None] * np.ones_like(r)
        lam_r = Jg / (lam_t * lam_zz)                                   # Eq. (25)
        lam_er = lam_r / lam_g[..., 0]
        lam_et = lam_t / lam_g[..., 1]
        lam_ez = lam_zz / lam_g[..., 2]
        # ---- extra stresses (layer-wise material)
        s_rr_h = np.empty_like(r); s_tt_h = np.empty_like(r); s_zz_h = np.empty_like(r)
        for j, k in enumerate(ids):
            a, b, c = goh_extra_stress(lam_er[:, j], lam_et[:, j], lam_ez[:, j], self.layers[k].material)
            s_rr_h[:, j], s_tt_h[:, j], s_zz_h[:, j] = a, b, c
        # ---- radial equilibrium  d sigma_rr / dr = (sigma_tt - sigma_rr) / r, integrated in R
        q = (s_tt_h - s_rr_h) * lam_r / r
        cumq = np.einsum("lij,blj->bli", C, q)
        totq = (q * wts[None]).sum(-1)
        s_rr_in = np.empty((B, nL))
        s_rr_in[:, 0] = -p
        for k in range(1, nL):
            s_rr_in[:, k] = s_rr_in[:, k - 1] + totq[:, k - 1]
        s_rr = s_rr_in[..., None] + cumq
        s_tt = s_rr + (s_tt_h - s_rr_h)
        s_zz = s_rr + (s_zz_h - s_rr_h)
        s_rr_outer = s_rr_in[:, -1] + totq[:, -1]      # = 0 at equilibrium (Eq. 27)
        dA = r * lam_r * wts[None]                       # r dr = r lam_r dR
        Fz = (s_zz * dA).sum((1, 2))                     # proportional to net axial force
        M = (s_tt * dA).sum(-1)                          # bending moment per layer (B, L)
        return dict(r=r, r_in=r_in, r_out=r_out, lam_r=lam_r, lam_t=lam_t, lam_z=lam_zz,
                    lam_er=lam_er, lam_et=lam_et, lam_ez=lam_ez, Jg=Jg,
                    sig_rr=s_rr, sig_tt=s_tt, sig_zz=s_zz,
                    residual_p=s_rr_outer, Fz=Fz, M=M, Lam=Lam, Phi=Phi, p=p)

    # ------------------------------------------------------------------ configurations
    def loaded(self, lam_g, p_kPa=None, lam_z=None):
        """Pressurised, axially stretched configuration B_t: solve Eq. (27) for r_i (batched)."""
        lam_g = np.asarray(lam_g, float)
        if lam_g.ndim == 3:
            lam_g = lam_g[None]
        B = lam_g.shape[0]
        p = self.p_kPa if p_kPa is None else p_kPa
        Lam = self.lam_z if lam_z is None else lam_z
        R_i = self.layers[0].R_i

        def resid(x):
            return self.evaluate(lam_g, x, Lam, 2.0 * np.pi, p)["residual_p"]

        r_i = bisect_vec(resid, np.full(B, 0.7 * R_i), np.full(B, 2.5 * R_i), n_iter=60)
        out = self.evaluate(lam_g, r_i, Lam, 2.0 * np.pi, p)
        out["r_i"] = r_i
        out["tau_w"] = wall_shear_stress(self.Q_ml_min, self.mu_cP, r_i)
        return out

    def unloaded(self, lam_g, x0=None):
        """Traction-free intact configuration B_a (p = 0, zero net axial force).  Single sample."""
        lam_g = np.asarray(lam_g, float)
        if lam_g.ndim == 3:
            lam_g = lam_g[None]
        assert lam_g.shape[0] == 1
        if x0 is None:
            x0 = np.array([self.layers[0].R_i, 1.0])

        def fun(x):
            o = self.evaluate(lam_g, x[0], x[1], 2.0 * np.pi, 0.0)
            return np.array([o["residual_p"][0], o["Fz"][0] / 100.0])

        sol = root(fun, x0, method="hybr", tol=1e-12)
        if not sol.success:
            raise RuntimeError("unloaded(): root finding failed: " + sol.message)
        out = self.evaluate(lam_g, sol.x[0], sol.x[1], 2.0 * np.pi, 0.0)
        out["r_i"] = sol.x[0]
        return out

    def _sector_equilibrium(self, lg, layer, Phi, x0):
        """For a fixed sector angle Phi solve p = 0 and F_z = 0 for (rho_i, Lambda) of one layer."""
        def fun(x):
            o = self.evaluate(lg, x[0], x[1], Phi, 0.0, layer_ids=[layer])
            return np.array([o["residual_p"][0], o["Fz"][0] / 100.0])
        sol = root(fun, x0, method="hybr", tol=1e-13)
        return sol.x, sol.success

    def opening_angle(self, lam_g, layer, phi_bracket=(0.05, 2.0 * np.pi), n_scan=24):
        """Cut-open configuration of a single layer: p = 0, zero axial force, zero bending moment.

        The problem is nearly degenerate in (rho_i, Phi) for thin sectors, so it is solved in a nested
        way: for each sector angle Phi the radial/axial equilibrium gives (rho_i, Lambda); the bending
        moment M(Phi) is then driven to zero with a bracketing scalar root finder.
        Returns dict with ``Phi`` (sector angle), ``alpha_deg`` (opening angle = pi - Phi/2, Fung's
        definition, 0 deg for a closed stress-free ring) and the stresses of the cut configuration.
        """
        from scipy.optimize import brentq
        lam_g = np.asarray(lam_g, float)
        if lam_g.ndim == 3:
            lam_g = lam_g[None]
        lg = lam_g[:, [layer]]
        R_i, R_o = self.layers[layer].R_i, self.layers[layer].R_o
        state = {"x": np.array([R_i, 1.0])}

        def moment(Phi):
            # arc-length-preserving initial guess: rho_i ~ R_i * 2 pi / Phi
            x0 = np.array([R_i * 2.0 * np.pi / Phi, state["x"][1]])
            x, ok = self._sector_equilibrium(lg, layer, Phi, x0)
            if not ok:
                x, ok = self._sector_equilibrium(lg, layer, Phi, state["x"])
            state["x"] = x
            return self.evaluate(lg, x[0], x[1], Phi, 0.0, layer_ids=[layer])["M"][0, 0]

        # scan for a sign change of M(Phi), starting from the closed ring
        phis = np.linspace(phi_bracket[1], phi_bracket[0], n_scan)
        m_prev, phi_prev = moment(phis[0]), phis[0]
        if abs(m_prev) < 1e-9:                       # stress-free closed ring
            Phi = phis[0]
        else:
            Phi = None
            for ph in phis[1:]:
                m = moment(ph)
                if np.sign(m) != np.sign(m_prev):
                    Phi = brentq(moment, ph, phi_prev, xtol=1e-12)
                    break
                m_prev, phi_prev = m, ph
            if Phi is None:
                raise RuntimeError("opening_angle(): no sign change of the bending moment found")
        moment(Phi)
        x = state["x"]
        out = self.evaluate(lg, x[0], x[1], Phi, 0.0, layer_ids=[layer])
        out["rho_i"] = x[0]
        out["Phi"] = Phi
        out["alpha_deg"] = np.rad2deg(np.pi - 0.5 * Phi)
        return out
