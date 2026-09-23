"""Expert growth law of Liu et al. (2019), Eq. (8)-(9), designed for residual stress generation.

The exponent beta and the pause threshold h0 are not given in the RL paper; beta = 1 and
h0 = 0.5 are used as defaults (see README).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Liu2019Law:
    alpha: tuple = (1e-3, 1e-3, 0.0, 1e-3)   # alpha01..alpha04 [1/kPa]
    beta: float = 1.0
    T: float = 1.0                            # days
    h0: float = 0.5

    def rate(self, sig_rr, sig_tt, sig_zz):
        a1, a2, a3, a4 = self.alpha
        sig_rr, sig_tt, sig_zz = (np.asarray(x, float) for x in (sig_rr, sig_tt, sig_zz))
        rate = np.zeros(sig_tt.shape + (3,))
        rate[..., 0] = (a1 * np.abs(sig_tt) + a2 * np.abs(sig_zz)) ** self.beta / self.T
        rate[..., 1] = (a3 * np.abs(sig_rr) + a4 * np.abs(sig_zz)) ** self.beta / self.T
        return rate

    @staticmethod
    def homogeneity(mean_now, mean_0):
        """Eq. (9): h = (mean_M(t) - mean_A(t)) / (mean_M(0) - mean_A(0)) for a (…, 2) layer array."""
        return (mean_now[..., 0] - mean_now[..., 1]) / (mean_0[..., 0] - mean_0[..., 1])
