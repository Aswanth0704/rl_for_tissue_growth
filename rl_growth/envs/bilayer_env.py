"""Example 2 environment: distributed growth policy for one layer of the bi-layer aorta (Section 3.2).

During training the agent controls growth at a single integration point (Loc 3 by default) of
one layer while Fg = I everywhere else.  Only circumferential and axial growth are allowed
(9 actions).  The distance metric weights tau_w by 100 (media) or 33.3 (adventitia).
"""
import numpy as np

from ..mechanics import BilayerAorta
from .base import GrowthEnvBase, make_action_table


class BilayerGrowthEnv(GrowthEnvBase):
    def __init__(self, aorta=None, layer=0, loc=2, g=0.01, lam_g_range=(0.9, 1.1),
                 grow_dims=("t", "z"), tau_weight=None, s_h=None,
                 beta1=-40.0, beta2=150.0, r_max=80.0, term_thresholds=(200.0, 200.0, 20.0),
                 seed=0, admissible_starts=True):
        self.aorta = aorta if aorta is not None else BilayerAorta()
        self.layer, self.loc = layer, loc
        self.g = g
        self.lam_g_range = lam_g_range
        self.grow_dims = grow_dims
        self.action_table = make_action_table(grow_dims, g)
        self.beta1, self.beta2, self.r_max = beta1, beta2, r_max
        self.term_thresholds = np.asarray(term_thresholds, float)
        self.rng = np.random.default_rng(seed)
        if tau_weight is None:
            tau_weight = 100.0 if layer == 0 else 33.3
        self.dist_weights = np.array([1.0, 1.0, tau_weight])
        # homeostatic state: transmural mean stresses of the layer before growth + tau_w,h
        if s_h is None:
            o = self.aorta.loaded(self.aorta.identity_growth())
            m_tt = self.aorta.layer_mean(o["sig_tt"])[0, layer]
            m_zz = self.aorta.layer_mean(o["sig_zz"])[0, layer]
            s_h = (m_tt, m_zz, o["tau_w"][0])
        self.s_h = np.asarray(s_h, float)
        n = int(round((lam_g_range[1] - lam_g_range[0]) / g)) + 1
        self.grid_axis = lam_g_range[0] + g * np.arange(n)
        self.grid_shape = (n, n)
        self.lam_g = None      # (n, 3) growth at the controlled point (r component stays 1)
        self.obs = None
        self._start_pool = None
        if admissible_starts:
            lam = self.grid_states()
            ok = ~self.is_terminal(self._stress_state(lam))
            self._start_pool = lam[ok]

    # ------------------------------------------------------------ mechanics
    def full_growth(self, lam_g):
        lam_g = np.atleast_2d(lam_g)
        full = self.aorta.identity_growth(lam_g.shape[0])
        full[:, self.layer, self.loc, :] = lam_g
        return full

    def _stress_state(self, lam_g):
        o = self.aorta.loaded(self.full_growth(lam_g))
        return np.stack([o["sig_tt"][:, self.layer, self.loc],
                         o["sig_zz"][:, self.layer, self.loc],
                         o["tau_w"]], axis=1)

    # ------------------------------------------------------------ episodes
    def sample_lam_g(self, n):
        if self._start_pool is not None:
            return self._start_pool[self.rng.integers(0, len(self._start_pool), n)].copy()
        idx = self.rng.integers(0, len(self.grid_axis), size=(n, 2))
        lam = np.ones((n, 3))
        lam[:, 1:] = self.grid_axis[idx]
        return lam

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
        g = np.stack(np.meshgrid(self.grid_axis, self.grid_axis, indexing="ij"), -1).reshape(-1, 2)
        lam = np.ones((g.shape[0], 3))
        lam[:, 1:] = g
        return lam

    def grid_transitions(self):
        n = len(self.grid_axis)
        idx = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
        deltas = np.rint(self.action_table[:, 1:] / self.g).astype(int)
        nxt = np.clip(idx[:, None, :] + deltas[None], 0, n - 1)
        return np.ravel_multi_index((nxt[..., 0], nxt[..., 1]), self.grid_shape)

    def build_mdp(self):
        lam = self.grid_states()
        s = self._stress_state(lam)
        return lam, s, self.reward(s), self.grid_transitions()
