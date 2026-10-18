# Reinforcement Learning growth laws for arteries

A reimplementation of

*M. Liu, L. Liang, H. Dong, W. Sun, R. L. Gleason, *Constructing growth evolution laws of arteries via reinforcement learning, J. Mech. Phys. Solids (2022).*


## Results

The figure below is the Example 1 output (hypertensive growth of the murine infrarenal abdominal aorta over 28 days), with the DQN trained for the paper's 30 000 episodes.

![Example 1: hypertensive growth of the murine IAA](results/example1/example1_hypertension.png)

## Running

Install the dependencies into whatever conda environment using the requirements.txt and run the notebook:

```bash
pip install -r requirements.txt
jupyter notebook rl_growth_dqn.ipynb
```

