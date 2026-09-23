"""Growth policies pi_g(s) -> action index that can be deployed in new mechanical environments."""
import numpy as np
from scipy.spatial import cKDTree


class NearestStatePolicy:
    """Tabular (DP) policy executed in a continuous stress space by nearest-neighbour lookup
    with the (weighted) Euclidean distance of Eq. (12)."""

    def __init__(self, stress_states, policy, dist_weights=(1.0, 1.0, 1.0)):
        self.w = np.asarray(dist_weights, float)
        self.tree = cKDTree(np.asarray(stress_states) * self.w)
        self.policy = np.asarray(policy, int)

    def __call__(self, s):
        _, idx = self.tree.query(np.atleast_2d(s) * self.w)
        return self.policy[idx]


class QNetworkPolicy:
    """Greedy policy of a trained DQN agent (Eq. 18).

    ``no_growth_action`` / ``quiescence_tol``: if the Q value of the "no growth" action is within
    ``quiescence_tol * |max Q|`` of the best action it is chosen instead.  Near homeostasis the
    clipped reward makes all actions (almost) equally good and the plain argmax becomes a random
    walk; this is the network analogue of the tie-breaking used in policy iteration.
    """

    def __init__(self, agent, no_growth_action=None, quiescence_tol=3e-3):
        self.agent = agent
        self.a0 = no_growth_action
        self.tol = quiescence_tol

    def __call__(self, s):
        q = self.agent.q_values(np.atleast_2d(s))
        a = q.argmax(1)
        if self.a0 is not None and self.tol > 0:
            best = q[np.arange(len(q)), a]
            quiet = q[:, self.a0] >= best - self.tol * np.abs(best)
            a = np.where(quiet, self.a0, a)
        return a
