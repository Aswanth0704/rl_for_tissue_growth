"""Example 2: growth-induced residual stresses in a bi-layer human ascending aorta (Section 3.2)."""
import json
import os
import time
import numpy as np

from ..envs import BilayerGrowthEnv
from ..mechanics import BilayerAorta
from ..rl import policy_iteration, NearestStatePolicy, QNetworkPolicy, DQNConfig, DQNAgent, train_dqn
from ..baselines import TaberHumphreyLaw, Liu2019Law
from .. import plotting

LAYER_NAMES = ("media", "adventitia")


def simulate(aorta, growth_step, n_steps=20, dt=0.2, lam_g0=None):
    """Collaborative growth of all integration points.

    growth_step(out, lam_g, step) -> (2, 5, 3) increments given the loaded-state dict ``out``.
    Returns dict(t, lam_g (T+1, 2, 5, 3), sig_tt/sig_zz/r (T+1, 2, 5), r_i, tau_w).
    """
    lam_g = aorta.identity_growth() if lam_g0 is None else np.array(lam_g0, float)[None]
    hist = {k: [] for k in ("t", "lam_g", "sig_rr", "sig_tt", "sig_zz", "r", "r_i", "r_out", "tau_w")}
    for n in range(n_steps + 1):
        out = aorta.loaded(lam_g)
        hist["t"].append(n * dt); hist["lam_g"].append(lam_g[0].copy())
        for k in ("sig_rr", "sig_tt", "sig_zz", "r", "r_i", "r_out", "tau_w"):
            hist[k].append(out[k][0])
        if n < n_steps:
            lam_g = lam_g + growth_step(out, lam_g, n)[None]
    return {k: np.asarray(v) for k, v in hist.items()}


def rl_growth_step(policies, action_tables):
    """policies[k](states (5,3)) -> action ids for layer k."""
    def step(out, lam_g, n):
        d = np.zeros((2, 5, 3))
        for k, pol in policies.items():
            s = np.stack([out["sig_tt"][0, k], out["sig_zz"][0, k], np.full(5, out["tau_w"][0])], 1)
            d[k] = action_tables[k][pol(s)]
        return d
    return step


def taber_humphrey_step(laws, dt):
    def step(out, lam_g, n):
        d = np.zeros((2, 5, 3))
        for k, law in enumerate(laws):
            d[k] = law.rate(out["sig_tt"][0, k], np.full(5, out["tau_w"][0])) * dt
        return d
    return step


def liu2019_step(law, aorta, mean_0, dt):
    def step(out, lam_g, n):
        mean_now = np.stack([aorta.layer_mean(out["sig_tt"])[0], aorta.layer_mean(out["sig_zz"])[0]], 0)  # (2 comp, 2 layers)
        h_t = law.homogeneity(mean_now[0], mean_0[0])
        h_z = law.homogeneity(mean_now[1], mean_0[1])
        if h_t < law.h0 and h_z < law.h0:
            return np.zeros((2, 5, 3))
        return law.rate(out["sig_rr"][0], out["sig_tt"][0], out["sig_zz"][0]) * dt
    return step


def postprocess(aorta, hist):
    """Loaded stresses, residual stresses in B_a and opening angles for the final growth state."""
    lam_g = hist["lam_g"][-1]
    loaded = aorta.loaded(lam_g)
    unl = aorta.unloaded(lam_g)
    angles = [aorta.opening_angle(lam_g, k)["alpha_deg"] for k in range(2)]
    return dict(loaded=dict(r=loaded["r"][0], sig_tt=loaded["sig_tt"][0], sig_zz=loaded["sig_zz"][0]),
                residual=dict(r=unl["r"][0], sig_tt=unl["sig_tt"][0], sig_zz=unl["sig_zz"][0]),
                opening_angle_deg=angles, r_i_loaded=float(loaded["r_i"][0]), r_out_loaded=float(loaded["r_out"][0]),
                r_i_unloaded=float(unl["r_i"]), lam_z_unloaded=float(unl["Lam"][0]))


def run(outdir="results/example2", methods=("dp", "dqn", "th", "liu"), dqn_episodes=3600, dqn_max_steps=500,
        dqn_n_envs=64, dqn_cfg=None, n_steps=20, dt=0.2, gamma=0.99, seed=0, load_dqn=None, liu_h0=0.5, quiescence_tol=3e-3):
    os.makedirs(outdir, exist_ok=True)
    aorta = BilayerAorta()
    envs = [BilayerGrowthEnv(aorta, layer=k, seed=seed) for k in range(2)]
    for k, env in enumerate(envs):
        print(f"[{LAYER_NAMES[k]}] homeostatic state s_h = {np.round(env.s_h, 3)}, tau weight = {env.dist_weights[2]}")
    action_tables = {k: envs[k].action_table for k in range(2)}
    histories, summary, info = {}, {}, {}

    if "dp" in methods:
        policies = {}
        for k, env in enumerate(envs):
            t0 = time.time()
            lam, s, r, nxt = env.build_mdp()
            policy, V = policy_iteration(nxt, r, gamma=gamma, verbose=False, preferred_action=env.no_growth_action)
            print(f"[DP {LAYER_NAMES[k]}] {len(lam)} states, min distance {env.distance(s).min():.3f}, {time.time() - t0:.1f}s")
            np.savez(os.path.join(outdir, f"dp_policy_{LAYER_NAMES[k]}.npz"), lam_g=lam, stress=s, reward=r,
                     policy=policy, value=V, action_table=env.action_table, dist_weights=env.dist_weights)
            policies[k] = NearestStatePolicy(s, policy, env.dist_weights)
        histories["Dynamic programming"] = simulate(aorta, rl_growth_step(policies, action_tables), n_steps, dt)

    if "dqn" in methods:
        policies = {}
        for k, env in enumerate(envs):
            cfg = dqn_cfg or DQNConfig(seed=seed)
            path = os.path.join(outdir, f"dqn_agent_{LAYER_NAMES[k]}.pt")
            if load_dqn:
                agent = DQNAgent.load(os.path.join(load_dqn, f"dqn_agent_{LAYER_NAMES[k]}.pt"))
            else:
                agent = DQNAgent(3, env.n_actions, obs_offset=env.s_h,
                                 obs_scale=(50.0, 50.0, 50.0 / env.dist_weights[2]), cfg=cfg)
                t0 = time.time()
                hist = train_dqn(env, agent, n_episodes=dqn_episodes, max_steps=dqn_max_steps, n_envs=dqn_n_envs)
                info[f"dqn_time_s_{LAYER_NAMES[k]}"] = time.time() - t0
                np.savez(os.path.join(outdir, f"dqn_training_{LAYER_NAMES[k]}.npz"), **hist)
                plotting.plot_training(hist, os.path.join(outdir, f"dqn_training_{LAYER_NAMES[k]}.png"),
                                       title=f"DQN training ({LAYER_NAMES[k]})")
                agent.save(path)
            policies[k] = QNetworkPolicy(agent, env.no_growth_action, quiescence_tol)
        histories["Deep Q network"] = simulate(aorta, rl_growth_step(policies, action_tables), n_steps, dt)

    if "th" in methods:
        laws = [TaberHumphreyLaw(sig_tt_h=envs[k].s_h[0], tau_h=envs[k].s_h[2]) for k in range(2)]
        histories["Taber & Humphrey (2001)"] = simulate(aorta, taber_humphrey_step(laws, dt), n_steps, dt)

    if "liu" in methods:
        o0 = aorta.loaded(aorta.identity_growth())
        mean_0 = np.stack([aorta.layer_mean(o0["sig_tt"])[0], aorta.layer_mean(o0["sig_zz"])[0]], 0)
        law = Liu2019Law(h0=liu_h0)
        histories["Liu et al. (2019)"] = simulate(aorta, liu2019_step(law, aorta, mean_0, dt), n_steps, dt)

    # ---- post-processing
    before = postprocess(aorta, {"lam_g": aorta.identity_growth()})
    loaded_plots = {"Before growth": before["loaded"]}
    residual_plots = {}
    for name, h in histories.items():
        pp = postprocess(aorta, h)
        summary[name] = dict(opening_angle_media_deg=pp["opening_angle_deg"][0],
                             opening_angle_adventitia_deg=pp["opening_angle_deg"][1],
                             r_i_loaded=pp["r_i_loaded"], r_out_loaded=pp["r_out_loaded"],
                             mean_sig_tt_media=float(aorta.layer_mean(pp["loaded"]["sig_tt"][None])[0, 0]),
                             mean_sig_tt_adventitia=float(aorta.layer_mean(pp["loaded"]["sig_tt"][None])[0, 1]),
                             final_lam_g=h["lam_g"][-1].tolist())
        loaded_plots[name] = pp["loaded"]
        residual_plots[name] = pp["residual"]
        tag = name.split()[0].lower().replace("&", "and")
        np.savez(os.path.join(outdir, f"sim_{tag}.npz"), **h, r_res=pp["residual"]["r"],
                 sig_tt_res=pp["residual"]["sig_tt"], sig_zz_res=pp["residual"]["sig_zz"])
        print(f"{name:26s} opening angles (media, adventitia) = ({pp['opening_angle_deg'][0]:.2f}, "
              f"{pp['opening_angle_deg'][1]:.2f}) deg ; loaded r_i = {pp['r_i_loaded']:.3f}, r_o = {pp['r_out_loaded']:.3f} mm")
    plotting.plot_example2_stress(loaded_plots, os.path.join(outdir, "example2_loaded_stress.png"),
                                  "Loaded configuration B_t (Fig. 11)")
    if residual_plots:
        plotting.plot_example2_stress(residual_plots, os.path.join(outdir, "example2_residual_stress.png"),
                                      "Residually stressed configuration B_a (Fig. 12)")
        plotting.plot_example2_growth(histories, 0, os.path.join(outdir, "example2_growth_loc1.png"))
        plotting.plot_example2_growth(histories, 4, os.path.join(outdir, "example2_growth_loc5.png"))
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump({"summary": summary, "info": info}, f, indent=2)
    return histories, summary
