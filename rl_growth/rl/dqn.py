"""Double deep Q network (Section 2.4.2) in PyTorch.

* prediction network Q_phi and slowly updated target network Q_xi
  (xi <- eta phi + (1 - eta) xi after every gradient step)
* epsilon-greedy exploration, experience replay, Adam, MSE loss on the double-DQN target
      y = r + gamma * Q_xi(s', argmax_a Q_phi(s', a))
* environments are *batched*: ``n_envs`` independent copies of the mechanical environment
  are stepped together (purely a throughput device; the algorithm is unchanged).
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(64, 64)):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.cap = capacity
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros(capacity, np.int64)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.ptr, self.size = 0, 0

    def add(self, obs, act, rew, next_obs, done):
        n = len(act)
        idx = (self.ptr + np.arange(n)) % self.cap
        self.obs[idx], self.act[idx], self.rew[idx] = obs, act, rew
        self.next_obs[idx], self.done[idx] = next_obs, done
        self.ptr = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.size, batch)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx]


@dataclass
class DQNConfig:
    hidden: tuple = (64, 64)
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: float = 64
    buffer_size: int = 1_000_000
    tau: float = 1e-4              # target smoothing factor eta (paper: 1e-4)
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_frac: float = 0.3    # fraction of the planned env steps over which eps decays linearly
    learning_starts: int = 1000
    updates_per_step: int = 1      # gradient updates per batched environment step
    reward_scale: float = 1.0 / 80.0
    grad_clip: float = 10.0
    seed: int = 0
    device: str = "cpu"


class DQNAgent:
    def __init__(self, obs_dim, n_actions, obs_offset, obs_scale, cfg: DQNConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.dev = torch.device(cfg.device)
        if self.dev.type == "cpu":
            torch.set_num_threads(1)      # tiny MLP: intra-op threading only adds overhead
        self.obs_offset = np.asarray(obs_offset, np.float32)
        self.obs_scale = np.asarray(obs_scale, np.float32)
        self.n_actions = n_actions
        self.q = MLP(obs_dim, n_actions, cfg.hidden).to(self.dev)
        self.q_target = MLP(obs_dim, n_actions, cfg.hidden).to(self.dev)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim)
        self.n_updates = 0

    # ------------------------------------------------------------ inference
    def normalize(self, obs):
        return (np.asarray(obs, np.float32) - self.obs_offset) / self.obs_scale

    @torch.no_grad()
    def q_values(self, obs):
        x = torch.as_tensor(self.normalize(obs), device=self.dev)
        return self.q(x).cpu().numpy()

    def act(self, obs, eps=0.0):
        obs = np.atleast_2d(obs)
        greedy = self.q_values(obs).argmax(1)
        if eps <= 0.0:
            return greedy
        explore = self.rng.random(len(obs)) < eps
        rand = self.rng.integers(0, self.n_actions, len(obs))
        return np.where(explore, rand, greedy)

    # ------------------------------------------------------------ learning
    def update(self):
        cfg = self.cfg
        o, a, r, o2, d = self.buffer.sample(int(cfg.batch_size), self.rng)
        o = torch.as_tensor(self.normalize(o), device=self.dev)
        o2 = torch.as_tensor(self.normalize(o2), device=self.dev)
        a = torch.as_tensor(a, device=self.dev)
        r = torch.as_tensor(r * cfg.reward_scale, device=self.dev)
        d = torch.as_tensor(d, device=self.dev)
        with torch.no_grad():
            a2 = self.q(o2).argmax(1, keepdim=True)                  # action chosen by prediction net
            q2 = self.q_target(o2).gather(1, a2).squeeze(1)          # evaluated by target net
            y = r + cfg.gamma * (1.0 - d) * q2
        q = self.q(o).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.mse_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(self.q.parameters(), cfg.grad_clip)
        self.opt.step()
        with torch.no_grad():
            for pt, p in zip(self.q_target.parameters(), self.q.parameters()):
                pt.mul_(1.0 - cfg.tau).add_(cfg.tau * p)
        self.n_updates += 1
        return float(loss)

    # ------------------------------------------------------------ io
    def save(self, path):
        torch.save({"q": self.q.state_dict(), "q_target": self.q_target.state_dict(),
                    "cfg": asdict(self.cfg), "obs_offset": self.obs_offset, "obs_scale": self.obs_scale,
                    "n_actions": self.n_actions}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, weights_only=False)
        cfg = DQNConfig(**ck["cfg"])
        cfg.hidden = tuple(cfg.hidden)
        agent = cls(len(ck["obs_offset"]), ck["n_actions"], ck["obs_offset"], ck["obs_scale"], cfg)
        agent.q.load_state_dict(ck["q"])
        agent.q_target.load_state_dict(ck["q_target"])
        return agent


def train_dqn(env, agent, n_episodes, max_steps=500, n_envs=16, log_every=200, verbose=True):
    """Train ``agent`` on a batched growth environment.

    An episode starts from a random discrete stress state and ends after ``max_steps`` or when
    the state leaves the admissible region (env.is_terminal).  Returns a history dict.
    """
    cfg = agent.cfg
    planned_steps = n_episodes * max_steps
    decay_steps = max(1, int(cfg.eps_decay_frac * planned_steps))
    obs = env.reset(n_envs)
    ep_ret = np.zeros(n_envs)
    ep_len = np.zeros(n_envs, int)
    ep_final_dist = np.zeros(n_envs)
    hist = {"episode": [], "return": [], "length": [], "final_distance": [], "eps": [], "loss": [], "steps": []}
    total_steps, episodes_done, last_loss = 0, 0, np.nan
    t0 = time.time()
    while episodes_done < n_episodes:
        eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * max(0.0, 1.0 - total_steps / decay_steps)
        a = agent.act(obs, eps)
        next_obs, r, done, _ = env.step(a)
        ep_len += 1
        trunc = ep_len >= max_steps
        agent.buffer.add(obs, a, r, next_obs, done.astype(np.float32))
        ep_ret += r
        total_steps += n_envs
        if agent.buffer.size >= cfg.learning_starts:
            for _ in range(cfg.updates_per_step):
                last_loss = agent.update()
        finished = done | trunc
        if finished.any():
            dist = env.distance(next_obs)
            for i in np.flatnonzero(finished):
                episodes_done += 1
                hist["episode"].append(episodes_done)
                hist["return"].append(ep_ret[i])
                hist["length"].append(ep_len[i])
                hist["final_distance"].append(dist[i])
                hist["eps"].append(eps)
                hist["loss"].append(last_loss)
                hist["steps"].append(total_steps)
                if verbose and episodes_done % log_every == 0:
                    k = max(1, log_every)
                    print(f"  ep {episodes_done:6d} | steps {total_steps:9d} | eps {eps:.3f} | "
                          f"return {np.mean(hist['return'][-k:]):8.1f} | len {np.mean(hist['length'][-k:]):6.1f} | "
                          f"final d {np.mean(hist['final_distance'][-k:]):7.2f} | loss {last_loss:.4f} | "
                          f"{time.time() - t0:7.0f}s")
            ep_ret[finished] = 0.0
            ep_len[finished] = 0
            next_obs = env.reset_where(finished)
        obs = next_obs
    return {k: np.asarray(v) for k, v in hist.items()}
