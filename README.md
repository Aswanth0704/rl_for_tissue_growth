# RL growth laws for arteries, re-implemented in PyTorch

A re-implementation of

> M. Liu, L. Liang, H. Dong, W. Sun, R. L. Gleason, *Constructing growth evolution laws of arteries via
> reinforcement learning*, J. Mech. Phys. Solids (2022).

The original work was done in MATLAB with MDPtoolbox and the Reinforcement Learning Toolbox. This
version uses NumPy and SciPy for the mechanics and PyTorch for the deep Q network.

```
rl_growth/
  units.py                     mmHg -> kPa, Poiseuille wall shear stress (Eq. 4)
  mechanics/fourfiber.py       four-fiber family model + remodelling (Eqs. 19-20)
  mechanics/goh.py             Gasser-Ogden-Holzapfel model (Eq. 23)
  mechanics/thin_wall.py       thin-walled IAA: Laplace equilibrium, Eqs. 21-22 (batched bisection)
  mechanics/thick_wall.py      bi-layer thick wall: Eqs. 24-27, 5-pt Gauss quadrature, unloading, opening angles
  envs/iaa_env.py              Example-1 MDP (27 actions, state [s_tt, s_zz, tau_w], reward Eq. 13)
  envs/bilayer_env.py          Example-2 MDP per layer (9 actions, weighted distance)
  rl/dp.py                     policy iteration on the discrete growth grid (Eqs. 15-16)
  rl/dqn.py                    double DQN with soft target update (Eq. 17), batched environments
  rl/policies.py               nearest-state lookup of tabular policies, greedy Q policy (Eq. 18)
  baselines/                   Taber & Humphrey 2001 (Eq. 7), Liu et al. 2019 (Eqs. 8-9)
  data/bersi2017.py            approximate experimental values digitised from Figs. 6 and 8
  experiments/                 Example 1 (hypertension) and Example 2 (residual stress) drivers
  plotting.py                  figures mirroring Figs. 6-8 and 11-14
scripts/run_example1.py, scripts/run_example2.py
tests/test_mechanics.py        checks against numbers quoted in the paper
```

## Results at a glance

The figure below is the Example 1 output (hypertensive growth of the murine infrarenal abdominal
aorta over 28 days) from `scripts/run_example1.py --methods dp dqn th`, with the DQN trained for
the paper's 30 000 episodes.

![Example 1: hypertensive growth of the murine IAA](results/example1/example1_hypertension.png)

The panels correspond to three figures of the paper, so they can be compared directly with the
[version of record](https://www.sciencedirect.com/science/article/pii/S002250962200223X):

| Panels here | Paper |
|---|---|
| (a) pressure, (b) unloaded thickness, (c) inner radius, (d) in vivo axial stretch | Fig. 6 (a)-(d) |
| (e) radial, (f) circumferential, (g) axial growth stretch | Fig. 7 (a)-(c) |
| (h) circumferential stress, (i) axial stress, (j) wall shear stress | Fig. 8 (a)-(c) |

Two things differ from the paper's plots. The paper also draws the constrained-mixture curve of
Latorre et al. (2019), which is not re-implemented here, and its Fig. 6 shows experimental error
bars where this figure shows only the digitised mean points.


