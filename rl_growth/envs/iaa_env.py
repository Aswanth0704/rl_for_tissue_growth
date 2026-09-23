"""Example 1 environment: homogeneous growth of the thin-walled murine IAA (Section 3.1)."""
import numpy as np

from ..mechanics import ThinWallArtery
from ..units import mmhg_to_kpa
from .base import GrowthEnvBase, make_action_table


class IAAGrowthEnv(GrowthEnvBase):
    def __init__(self, artery=None, p_mmHg=111.6, s_h=(221.0, 226.2, 26.8), g=0.01,
                 lam_g_range=(0.8, 1.2), grow_dims=("r", "t", "z"),
                 beta1=-40.0, beta2=150.0, r_max=80.0, term_thresholds=(200.0, 200.0, 20.0),
                 seed=0, admissible_starts=True):
        self.artery = artery if artery is not None else ThinWallArtery()
        self.p_kPa = mmhg_to_kpa(p_mmHg)
        self.s_h = np.asarray(s_h, float)
        self.dist_weights = np.ones(3)
        self.g = g
        self.lam_g_range = lam_g_range
        self.action_table = make_action_table(grow_dims, g)
        self.beta1, self.beta2, self.r_max = beta1, beta2, r_max
        self.term_thresholds = np.asarray(term_thresholds, float)
        self.rng = np.random.default_rng(seed)
        # discrete state grid used for dynamic programming (Section 2.4.1)
        n = int(round((lam_g_range[1] - lam_g_range[0]) / g)) + 1
        self.grid_axis = lam_g_range[0] + g * np.arange(n)
        self.grid_shape = (n, n, n)
        self.lam_g = None
        self.obs = None
        # episodes start from a random discrete state; by default only from states inside the
        # termination thresholds (states outside would end the episode after a single step)
        self._start_pool = None
        if admissible_starts:
            lam = self.grid_states()
            ok = ~self.is_terminal(self._stress_state(lam))
            self._start_pool = lam[ok]

    # ------------------------------------------------------------ mechanics
    def _stress_state(self, lam_g, p_kPa=None):
        p = self.p_kPa if p_kPa is None else p_kPa
        return self.artery.stress_state(lam_g, p)

    # ------------------------------------------------------------ episodes
    def sample_lam_g(self, n):
        if self._start_pool is not None:
            return self._start_pool[self.rng.integers(0, len(self._start_pool), n)].copy()
        idx = self.rng.integers(0, len(self.grid_axis), size=(n, 3))
        return self.grid_axis[idx]

    def reset(self, n=1, lam_g=None):
        self.lam_g = self.sample_lam_g(n) if lam_g is None else np.atleast_2d(np.asarray(lam_g, float)).copy()
        self.obs = self._stress_state(self.lam_g)
        return self.obs

    def reset_where(self, mask):
        mask = np.asarray(mask, bool)
        if mask.any():
            self.lam_g[mask] = self.sample_lam_g(int(mask.sum()))
            self.obs[mask] = self._stress_state(self.lam_g[mask])
        return self.obs

    # ------------------------------------------------------------ dynamic programming
    def grid_states(self):
        """All discrete growth states (N, 3) in C order over (r, theta, z)."""
        g = np.stack(np.meshgrid(self.grid_axis, self.grid_axis, self.grid_axis, indexing="ij"), -1)
        return g.reshape(-1, 3)

    def grid_transitions(self):
        """Deterministic transition table next_state[s, a] on the growth-stretch grid (clamped)."""
        n = len(self.grid_axis)
        idx = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 3)
        deltas = np.rint(self.action_table / self.g).astype(int)          # (A, 3)
        nxt = np.clip(idx[:, None, :] + deltas[None, :, :], 0, n - 1)     # (N, A, 3)
        return np.ravel_multi_index((nxt[..., 0], nxt[..., 1], nxt[..., 2]), self.grid_shape)

    def build_mdp(self):
        """Return (lam_g_states, stress_states, rewards, next_state) for policy iteration."""
        lam = self.grid_states()
        s = self._stress_state(lam)
        return lam, s, self.reward(s), self.grid_transitions()
