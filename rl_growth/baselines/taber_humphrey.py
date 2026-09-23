"""Expert growth law of Taber & Humphrey (2001), Eq. (7)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class TaberHumphreyLaw:
    sig_tt_h: float
    tau_h: float
    T_r: float = 0.3      # days
    T_t: float = 3.0
    T_tau: float = 5.0

    def rate(self, sig_tt, tau_w):
        """d lam_g / dt as an (n, 3) array for (r, theta, z)."""
        s = np.asarray(sig_tt, float) / self.sig_tt_h
        t = np.asarray(tau_w, float) / self.tau_h
        rate = np.zeros(np.broadcast(s, t).shape + (3,))
        rate[..., 0] = (s - 1.0) / self.T_r
        rate[..., 1] = (s - 1.0) / self.T_t + (t - 1.0) / self.T_tau
        return rate
