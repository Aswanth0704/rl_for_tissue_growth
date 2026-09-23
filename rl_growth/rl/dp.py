"""Dynamic programming on the discretised stress-state space (Section 2.4.1).

The transitions are deterministic, so P(s'|s,a) is represented by an integer table
``next_state[s, a]``.  Policy iteration alternates policy evaluation (Eq. 15) and policy
improvement (Eq. 16), following mdp_policy_iteration of MDPtoolbox.
"""
import numpy as np


def evaluate_policy(next_state, reward, policy, gamma, tol=1e-8, max_iter=100000, v0=None):
    """Iterative policy evaluation  v(s) <- R(s) + gamma v(next(s, pi(s)))."""
    nxt = next_state[np.arange(len(policy)), policy]
    v = np.zeros(len(reward)) if v0 is None else v0.copy()
    for it in range(max_iter):
        v_new = reward + gamma * v[nxt]
        delta = np.max(np.abs(v_new - v))
        v = v_new
        if delta < tol:
            break
    return v


def policy_iteration(next_state, reward, gamma=0.99, policy0=None, max_iter=200, eval_tol=1e-8, verbose=True,
                     preferred_action=None):
    """Return (policy, value).  ``next_state`` is (N, A) int, ``reward`` is (N,).

    ``preferred_action`` (e.g. the index of the "no growth" action) is used as the initial policy
    and wins ties in the improvement step; near homeostasis the clipped reward makes many actions
    equally good and this choice keeps the tissue quiescent instead of oscillating.
    """
    N, A = next_state.shape
    if policy0 is None:
        policy = np.zeros(N, int) if preferred_action is None else np.full(N, preferred_action, int)
    else:
        policy = policy0.copy()
    v = None
    for it in range(max_iter):
        v = evaluate_policy(next_state, reward, policy, gamma, tol=eval_tol, v0=v)
        q = v[next_state]                                    # (N, A): sum_s' P v(s')
        best = q.max(1)
        # keep the current action when it is already optimal (guarantees termination)
        keep = q[np.arange(N), policy] >= best - 1e-9
        new_policy = np.where(keep, policy, q.argmax(1))
        if preferred_action is not None:
            pref_ok = q[:, preferred_action] >= best - 1e-9
            new_policy = np.where(pref_ok, preferred_action, new_policy)
        n_changed = int((new_policy != policy).sum())
        if verbose:
            print(f"  policy iteration {it:3d}: {n_changed} actions changed, mean V = {v.mean():.3f}")
        policy = new_policy
        if n_changed == 0:
            break
    return policy, v


def value_iteration(next_state, reward, gamma=0.99, tol=1e-8, max_iter=100000):
    N, A = next_state.shape
    v = np.zeros(N)
    for _ in range(max_iter):
        v_new = reward + gamma * v[next_state].max(1)
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    return v[next_state].argmax(1), v
