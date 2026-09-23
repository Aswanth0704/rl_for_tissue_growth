#!/usr/bin/env python
"""Example 2 (growth-induced residual stress in a bi-layer human ascending aorta).

    python scripts/run_example2.py --methods dp th liu          # fast (~1 min)
    python scripts/run_example2.py --methods dp dqn th liu --dqn-episodes 3600   # paper setting
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl_growth.experiments import example2_residual_stress as ex2
from rl_growth.rl import DQNConfig

p = argparse.ArgumentParser()
p.add_argument("--outdir", default="results/example2")
p.add_argument("--methods", nargs="+", default=["dp", "dqn", "th", "liu"])
p.add_argument("--dqn-episodes", type=int, default=3600)
p.add_argument("--dqn-max-steps", type=int, default=500)
p.add_argument("--dqn-n-envs", type=int, default=64)
p.add_argument("--dqn-updates-per-step", type=int, default=4)
p.add_argument("--dqn-tau", type=float, default=1e-4)
p.add_argument("--dqn-lr", type=float, default=1e-3)
p.add_argument("--dqn-hidden", type=int, nargs="+", default=[64, 64])
p.add_argument("--dqn-eps-decay-frac", type=float, default=0.3)
p.add_argument("--load-dqn", default=None, help="directory with dqn_agent_media.pt / dqn_agent_adventitia.pt")
p.add_argument("--n-steps", type=int, default=20, help="growth steps (20 x 0.2 day = 4 days)")
p.add_argument("--dt", type=float, default=0.2)
p.add_argument("--liu-h0", type=float, default=0.5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--quiescence-tol", type=float, default=3e-3, help="DQN deployment: prefer no growth when its Q is within tol*|Qmax| of the best action (0 = plain argmax)")
a = p.parse_args()
cfg = DQNConfig(hidden=tuple(a.dqn_hidden), lr=a.dqn_lr, tau=a.dqn_tau, updates_per_step=a.dqn_updates_per_step,
                eps_decay_frac=a.dqn_eps_decay_frac, seed=a.seed)
ex2.run(a.outdir, tuple(a.methods), a.dqn_episodes, a.dqn_max_steps, a.dqn_n_envs, cfg, a.n_steps, a.dt,
        seed=a.seed, load_dqn=a.load_dqn, liu_h0=a.liu_h0, quiescence_tol=a.quiescence_tol)
