"""PyTorch/NumPy re-implementation of

    Liu, Liang, Dong, Sun, Gleason (2022)
    "Constructing growth evolution laws of arteries via reinforcement learning"
    Journal of the Mechanics and Physics of Solids.

Sub-packages
------------
mechanics   finite-growth kinematics + constitutive models (thin-wall four-fiber
            IAA model, bi-layer GOH thick-wall aorta model)
envs        Markov-decision-process formulation of arterial growth
rl          dynamic programming (policy iteration) and double DQN solvers
baselines   expert-prescribed growth laws (Taber & Humphrey 2001, Liu et al. 2019)
experiments end-to-end scripts reproducing Examples 1 and 2 of the paper
"""
__version__ = "0.1.0"
