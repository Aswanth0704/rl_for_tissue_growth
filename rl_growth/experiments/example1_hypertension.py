"""Example 1: hypertensive growth of the murine infrarenal abdominal aorta (Section 3.1)."""
import json
import os
import time
import numpy as np
from scipy.interpolate import PchipInterpolator

from ..data import bersi2017 as data
from ..envs import IAAGrowthEnv
from ..mechanics import ThinWallArtery
from ..rl import policy_iteration, NearestStatePolicy, QNetworkPolicy, DQNConfig, DQNAgent, train_dqn
from ..baselines import TaberHumphreyLaw
from ..units import mmhg_to_kpa
from .. import plotting


def pressure_interpolator():
    """Piecewise cubic (PCHIP) interpolation of the systolic pressure (Fig. 6a)."""
    return PchipInterpolator(data.days, data.systolic_pressure_mmHg)


def simulate(artery, growth_step, n_steps=140, dt=0.2, p_fn=None, lam_g0=(1.0, 1.0, 1.0)):
    """Explicit-Euler growth simulation (Section 2.1).

    growth_step(out) -> (3,) increment of lam_g for the current mechanical state ``out``
    (the dict returned by ThinWallArtery.solve, extended with 'state').
    """
    p_fn = pressure_interpolator() if p_fn is None else p_fn
    lam_g = np.array(lam_g0, float)[None]
    hist = {k: [] for k in ("t", "p_mmHg", "lam_gr", "lam_gt", "lam_gz", "sig_tt", "sig_zz", "tau_w",
                            "H_unloaded", "r_i", "lam_ez", "h")}
    for n in range(n_steps + 1):
        t = n * dt
        p_mmHg = float(p_fn(t))
        out = artery.solve(lam_g, mmhg_to_kpa(p_mmHg))
        out["state"] = np.stack([out["sig_tt"], out["sig_zz"], out["tau_w"]], 1)
        hist["t"].append(t); hist["p_mmHg"].append(p_mmHg)
        hist["lam_gr"].append(lam_g[0, 0]); hist["lam_gt"].append(lam_g[0, 1]); hist["lam_gz"].append(lam_g[0, 2])
        hist["sig_tt"].append(out["sig_tt"][0]); hist["sig_zz"].append(out["sig_zz"][0]); hist["tau_w"].append(out["tau_w"][0])
        hist["H_unloaded"].append(artery.H * lam_g[0, 0]); hist["r_i"].append(out["r_i"][0])
        hist["lam_ez"].append(out["lam_ez"][0]); hist["h"].append(out["h"][0])
        if n < n_steps:
            lam_g = lam_g + np.asarray(growth_step(out), float).reshape(1, 3)
    return {k: np.asarray(v) for k, v in hist.items()}


def rl_growth_step(policy, action_table):
    return lambda out: action_table[policy(out["state"])[0]]


def taber_humphrey_step(law, dt):
    return lambda out: law.rate(out["sig_tt"], out["tau_w"])[0] * dt


def run(outdir="results/example1", methods=("dp", "dqn", "th"), dqn_episodes=30000, dqn_max_steps=500,
        dqn_n_envs=64, dqn_cfg=None, n_steps=140, dt=0.2, gamma=0.99, seed=0, load_dqn=None, quiescence_tol=3e-3):
    os.makedirs(outdir, exist_ok=True)
    env = IAAGrowthEnv(seed=seed)
    artery = env.artery
    results, info = {}, {}

    if "dp" in methods:
        t0 = time.time()
        lam, s, r, nxt = env.build_mdp()
        print(f"[DP] {len(lam)} discrete states built in {time.time() - t0:.1f}s; "
              f"min distance to homeostasis = {env.distance(s).min():.2f}")
        policy, V = policy_iteration(nxt, r, gamma=gamma, preferred_action=env.no_growth_action)
        info["dp_time_s"] = time.time() - t0
        print(f"[DP] policy iteration done in {info['dp_time_s']:.1f}s")
        np.savez(os.path.join(outdir, "dp_policy.npz"), lam_g=lam, stress=s, reward=r, policy=policy, value=V,
                 action_table=env.action_table)
        pol = NearestStatePolicy(s, policy)
        results["Dynamic programming"] = simulate(artery, rl_growth_step(pol, env.action_table), n_steps, dt)

    if "dqn" in methods:
        cfg = dqn_cfg or DQNConfig(seed=seed)
        if load_dqn:
            agent = DQNAgent.load(load_dqn)
        else:
            agent = DQNAgent(3, env.n_actions, obs_offset=env.s_h, obs_scale=(100.0, 100.0, 10.0), cfg=cfg)
            t0 = time.time()
            hist = train_dqn(env, agent, n_episodes=dqn_episodes, max_steps=dqn_max_steps, n_envs=dqn_n_envs)
            info["dqn_time_s"] = time.time() - t0
            print(f"[DQN] training done in {info['dqn_time_s']:.0f}s")
            np.savez(os.path.join(outdir, "dqn_training.npz"), **hist)
            plotting.plot_training(hist, os.path.join(outdir, "dqn_training.png"))
            agent.save(os.path.join(outdir, "dqn_agent.pt"))
        pol = QNetworkPolicy(agent, env.no_growth_action, quiescence_tol)
        results["Deep Q network"] = simulate(artery, rl_growth_step(pol, env.action_table), n_steps, dt)

    if "th" in methods:
        law = TaberHumphreyLaw(sig_tt_h=env.s_h[0], tau_h=env.s_h[2])
        results["Taber & Humphrey (2001)"] = simulate(artery, taber_humphrey_step(law, dt), n_steps, dt)

    for name, h in results.items():
        tag = name.split()[0].lower().replace("&", "and")
        np.savez(os.path.join(outdir, f"sim_{tag}.npz"), **h)
        print(f"{name:26s} day {h['t'][-1]:.0f}: lam_g = ({h['lam_gr'][-1]:.3f}, {h['lam_gt'][-1]:.3f}, {h['lam_gz'][-1]:.3f}) "
              f"H = {h['H_unloaded'][-1]:.3f} mm  r_i = {h['r_i'][-1]:.3f} mm  lam_ez = {h['lam_ez'][-1]:.3f}  "
              f"s = ({h['sig_tt'][-1]:.1f}, {h['sig_zz'][-1]:.1f}, {h['tau_w'][-1]:.1f})")
    plotting.plot_example1(results, data, env.s_h, os.path.join(outdir, "example1_hypertension.png"))
    with open(os.path.join(outdir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return results, info
