"""
Part (e): Continuous Control with DDPG
=======================================
Deep Deterministic Policy Gradient for the Two-Link Reacher.
State: s_t = (x_target, y_target, θ1, θ2, dθ1, dθ2, x_end, y_end)
Action: continuous torques τ ∈ [-1, 1]^2
Training: 100 episodes.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
from reacher_env import ReacherEnv

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "Heriot_Watt_Thesis_Template-main", "Figures")
os.makedirs(IMAGES_DIR, exist_ok=True)


# =====================================================================
# Neural Networks
# =====================================================================
class Actor(nn.Module):
    """Deterministic policy network: s -> a."""
    def __init__(self, state_dim=8, action_dim=2, hidden=256, hidden2=None):
        super().__init__()
        hidden2 = hidden2 or hidden
        self.fc1 = nn.Linear(state_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)

        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Q-value network: (s, a) -> Q."""
    def __init__(self, state_dim=8, action_dim=2, hidden=256, hidden2=None):
        super().__init__()
        hidden2 = hidden2 or hidden
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


# =====================================================================
# Replay Buffer
# =====================================================================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =====================================================================
# Noise Processes
# =====================================================================
class OUNoise:
    """Ornstein-Uhlenbeck noise process."""
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1.0):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_init = sigma
        self.dt = dt
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(len(self.state))
        self.state += dx
        return self.state.copy()

    def decay(self, factor):
        self.sigma = max(0.01, self.sigma * factor)


class GaussianNoise:
    """Simple Gaussian noise."""
    def __init__(self, size, sigma=0.1):
        self.size = size
        self.sigma = sigma
        self.sigma_init = sigma

    def reset(self):
        pass

    def sample(self):
        return np.random.randn(self.size) * self.sigma

    def decay(self, factor):
        self.sigma = max(0.01, self.sigma * factor)


# =====================================================================
# Standard DDPG Agent
# =====================================================================
class DDPGAgent:
    def __init__(self, state_dim=8, action_dim=2, hidden=256, hidden2=None,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 batch_size=128, buffer_size=100000, noise_type="ou",
                 noise_sigma=0.2, reward_scale=1.0, grad_clip=1.0,
                 warmup_steps=0, updates_per_step=1, noise_decay=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.total_steps = 0
        self.noise_decay = noise_decay

        # Networks
        self.actor = Actor(state_dim, action_dim, hidden, hidden2)
        self.actor_target = Actor(state_dim, action_dim, hidden, hidden2)
        self.critic = Critic(state_dim, action_dim, hidden, hidden2)
        self.critic_target = Critic(state_dim, action_dim, hidden, hidden2)

        # Copy weights to targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Noise
        if noise_type == "ou":
            self.noise = OUNoise(action_dim, sigma=noise_sigma)
        else:
            self.noise = GaussianNoise(action_dim, sigma=noise_sigma)

    def get_extended_state(self, obs, env):
        """Build the 8-dim state: (x_t, y_t, θ1, θ2, dθ1, dθ2, x_end, y_end)."""
        x_t, y_t, θ1, θ2, dθ1, dθ2 = obs
        x_end = env.l1 * np.cos(θ1) + env.l2 * np.cos(θ1 + θ2)
        y_end = env.l1 * np.sin(θ1) + env.l2 * np.sin(θ1 + θ2)
        return np.array([x_t, y_t, θ1, θ2, dθ1, dθ2, x_end, y_end], dtype=np.float32)

    def select_action(self, state, add_noise=True):
        """Select action using actor + exploration noise."""
        if add_noise and self.total_steps < self.warmup_steps:
            return np.random.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

        state_t = torch.FloatTensor(state).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).numpy()
        self.actor.train()

        if add_noise:
            noise = self.noise.sample()
            action = action + noise
            action = np.clip(action, -1.0, 1.0)

        return action

    def update(self):
        """Standard DDPG update step."""
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1) * self.reward_scale
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        # --- Critic update ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            y = rewards_t + self.gamma * (1 - dones_t) * target_q

        current_q = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # --- Actor update ---
        actor_actions = self.actor(states_t)
        actor_loss = -self.critic(states_t, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # --- Soft update target networks ---
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def train_episode(self, env, max_steps=150):
        """Train for one episode."""
        obs, info = env.reset()
        state = self.get_extended_state(obs, env)
        self.noise.reset()

        total_reward = 0.0
        distances = []

        for step in range(max_steps):
            action = self.select_action(state, add_noise=True)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = self.get_extended_state(obs, env)
            done = terminated or truncated

            self.buffer.push(state, action, reward, next_state, float(done))
            self.total_steps += 1

            total_reward += reward
            distances.append(info["distance"])

            for _ in range(self.updates_per_step):
                self.update()

            state = next_state
            if done:
                break

        # Decay exploration noise each episode
        if self.noise_decay < 1.0:
            self.noise.decay(self.noise_decay)

        avg_dist = np.mean(distances)
        return total_reward, avg_dist, distances

    def evaluate(self, env, num_episodes=5):
        """Evaluate without noise."""
        returns = []
        avg_dists = []
        for _ in range(num_episodes):
            obs, info = env.reset()
            state = self.get_extended_state(obs, env)
            total_r = 0.0
            dists = []
            for _ in range(env.max_steps):
                action = self.select_action(state, add_noise=False)
                obs, r, term, trunc, info = env.step(action)
                total_r += r
                dists.append(info["distance"])
                state = self.get_extended_state(obs, env)
                if term or trunc:
                    break
            returns.append(total_r)
            avg_dists.append(np.mean(dists))
        return np.mean(returns), np.mean(avg_dists)


def train_ddpg(seed=42, num_episodes=100, target=(1.5, 0.0), noise_type="ou",
               noise_sigma=0.4, state_dim=8):
    """Train DDPG agent."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = ReacherEnv(dt=0.02, target=target)

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=2,
        hidden=64,
        hidden2=32,
        actor_lr=2e-3,
        critic_lr=4e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=64,
        buffer_size=50000,
        noise_type=noise_type,
        noise_sigma=noise_sigma,
        reward_scale=0.5,
        grad_clip=1.0,
        warmup_steps=32,
        updates_per_step=1,
        noise_decay=0.93,
    )

    episode_returns = []
    episode_errors = []

    for ep in range(num_episodes):
        total_r, avg_dist, _ = agent.train_episode(env)
        episode_returns.append(total_r)
        episode_errors.append(avg_dist)

        if (ep + 1) % 10 == 0:
            eval_return, eval_dist = agent.evaluate(env, num_episodes=3)
            print(f"Episode {ep+1}/{num_episodes} | "
                  f"Train Return: {total_r:.1f} | Train Dist: {avg_dist:.3f} | "
                  f"Eval Return: {eval_return:.1f} | Eval Dist: {eval_dist:.3f}")

    env.close()
    return agent, episode_returns, episode_errors


def main():
    print("=" * 60)
    print("Part (e): DDPG - Continuous Control")
    print("=" * 60)

    seeds = [42, 123, 456]
    all_returns = []
    all_errors = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        agent, returns, errors = train_ddpg(seed=seed, num_episodes=100)
        all_returns.append(returns)
        all_errors.append(errors)

    all_returns = np.array(all_returns)
    all_errors = np.array(all_errors)

    mean_returns = np.mean(all_returns, axis=0)
    std_returns = np.std(all_returns, axis=0)
    mean_errors = np.mean(all_errors, axis=0)
    std_errors = np.std(all_errors, axis=0)

    # Plot learning curves
    num_eps = all_returns.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    episodes = np.arange(1, num_eps + 1)

    axes[0].plot(episodes, mean_returns, 'b-')
    axes[0].fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns,
                          alpha=0.3)
    axes[0].set_title("DDPG: Episode Return")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].grid(True)

    axes[1].plot(episodes, mean_errors, 'r-')
    axes[1].fill_between(episodes, mean_errors - std_errors, mean_errors + std_errors,
                          alpha=0.3)
    axes[1].set_title("DDPG: Average Tracking Error")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Mean Distance")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "ddpg_100ep_learning_curves.pdf"))
    plt.close()
    print("\nPlot saved to ddpg_100ep_learning_curves.pdf")

    print(f"\nFinal performance (last 10 episodes, mean±std across seeds):")
    print(f"  Return: {np.mean(all_returns[:, -10:]):.2f} ± {np.std(all_returns[:, -10:]):.2f}")
    print(f"  Error:  {np.mean(all_errors[:, -10:]):.4f} ± {np.std(all_errors[:, -10:]):.4f}")


if __name__ == "__main__":
    main()
