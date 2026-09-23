"""Shared pieces of the growth MDP (Section 2.3)."""
import itertools
import numpy as np


def make_action_table(grow_dims, g):
    """Discrete growth actions: every combination of {+g, 0, -g} over ``grow_dims``.

    Returns an (N_A, 3) array of increments Delta lam_g = (r, theta, z); dimensions not in
    ``grow_dims`` are always 0.  With 3 growing dims N_A = 27, with 2 dims N_A = 9.
    """
    idx = {"r": 0, "t": 1, "z": 2}
    dims = [idx[d] for d in grow_dims]
    table = []
    for combo in itertools.product((-g, 0.0, g), repeat=len(dims)):
        row = np.zeros(3)
        for d, v in zip(dims, combo):
            row[d] = v
        table.append(row)
    return np.array(table)


def log_reward(d, beta1=-40.0, beta2=150.0, r_max=80.0, d_min=1e-8):
    """Eq. (13): f(d) = beta1 ln(d) + beta2, clipped to [-r_max, r_max]."""
    d = np.maximum(np.asarray(d, float), d_min)
    return np.clip(beta1 * np.log(d) + beta2, -r_max, r_max)


class GrowthEnvBase:
    """Common batched-environment interface.

    Sub-classes implement ``_stress_state(lam_g) -> (n, 3)`` and hold ``self.lam_g``.
    The state is s = [sigma_tt, sigma_zz, tau_w]; the reward is R(s) = f(d(s, s_h)).
    """
    s_h: np.ndarray            # homeostatic state (3,)
    dist_weights: np.ndarray   # weights of the (weighted) Euclidean distance (3,)
    action_table: np.ndarray   # (N_A, 3)
    term_thresholds: np.ndarray

    @property
    def n_actions(self):
        return self.action_table.shape[0]

    def distance(self, s):
        return np.linalg.norm((np.atleast_2d(s) - self.s_h) * self.dist_weights, axis=1)

    def reward(self, s):
        return log_reward(self.distance(s), self.beta1, self.beta2, self.r_max)

    def is_terminal(self, s):
        s = np.atleast_2d(s)
        return np.any(np.abs(s - self.s_h) >= self.term_thresholds, axis=1) | ~np.isfinite(s).all(1)

    @property
    def no_growth_action(self):
        return int(np.flatnonzero(np.all(self.action_table == 0.0, axis=1))[0])

    def step(self, actions):
        actions = np.asarray(actions, int)
        self.lam_g = self.lam_g + self.action_table[actions]
        s = self._stress_state(self.lam_g)
        bad = ~np.isfinite(s).all(1)          # equilibrium could not be solved -> failed episode
        if bad.any():
            s[bad] = self.s_h + 10.0 * self.term_thresholds
        self.obs = s
        return s, self.reward(s), self.is_terminal(s), {"lam_g": self.lam_g.copy(), "failed": bad}
