"""
Part (d): Discrete Action Control with N-Step SARSA
=====================================================
Discrete action space with neural network Q-function approximation.
Training: 100 episodes, static target (1.5, 0).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from reacher_env import ReacherEnv

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "Heriot_Watt_Thesis_Template-main", "Figures")
os.makedirs(IMAGES_DIR, exist_ok=True)


# =====================================================================
# Discrete action space
# =====================================================================
DISCRETE_ACTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),  (0, 0),  (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
NUM_ACTIONS = len(DISCRETE_ACTIONS)  # 9


def action_to_torque(action_idx, tau_max=1.0):
    """Convert discrete action index to continuous torque vector."""
    a = DISCRETE_ACTIONS[action_idx]
    return np.array([tau_max * a[0], tau_max * a[1]], dtype=np.float64)


# =====================================================================
# Neural Network Q-Function
# =====================================================================
class QNetwork(nn.Module):
    """Q-network: state -> Q-values for all actions."""
    def __init__(self, state_dim=6, num_actions=9, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, state):
        return self.net(state)


# =====================================================================
# N-Step SARSA with Neural Network
# =====================================================================
class NStepSARSA:
    """N-step SARSA with neural network function approximation."""

    def __init__(self, state_dim=6, num_actions=9, hidden=128, n_step=4,
                 lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=0.97):
        self.n_step = n_step
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.num_actions = num_actions

        self.q_net = QNetwork(state_dim, num_actions, hidden)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def q_value(self, state, action=None):
        """Compute Q(s, a) or Q(s) for all actions."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_vals = self.q_net(state_t).squeeze(0).numpy()
        if action is not None:
            return q_vals[action]
        return q_vals

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            q_vals = self.q_value(state)
            return np.argmax(q_vals)

    def train_episode(self, env):
        """Run one episode of N-step SARSA training."""
        obs, info = env.reset()
        state = obs.copy()
        action = self.select_action(state)

        states = [state]
        actions = [action]
        rewards = []

        T = 10**9
        t = 0
        total_reward = 0.0
        distances = []

        while True:
            if t < T:
                torque = action_to_torque(action)
                obs, reward, terminated, truncated, info = env.step(torque)
                total_reward += reward
                distances.append(info["distance"])
                rewards.append(reward)

                if terminated or truncated:
                    T = t + 1
                else:
                    next_state = obs.copy()
                    next_action = self.select_action(next_state)
                    states.append(next_state)
                    actions.append(next_action)

            tau_update = t - self.n_step + 1

            if tau_update >= 0:
                # Compute N-step return
                G = 0.0
                end_idx = min(tau_update + self.n_step, T)
                for i in range(tau_update, end_idx):
                    G += (self.gamma ** (i - tau_update)) * rewards[i]

                if tau_update + self.n_step < T:
                    s_end = states[tau_update + self.n_step]
                    a_end = actions[tau_update + self.n_step]
                    with torch.no_grad():
                        s_t = torch.FloatTensor(s_end).unsqueeze(0)
                        q_bootstrap = self.q_net(s_t)[0, a_end].item()
                    G += (self.gamma ** self.n_step) * q_bootstrap

                # SGD update
                s_update = torch.FloatTensor(states[tau_update]).unsqueeze(0)
                a_update = actions[tau_update]

                q_pred = self.q_net(s_update)[0, a_update]
                loss = (q_pred - G) ** 2

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                self.optimizer.step()

            if tau_update >= T - 1:
                break

            t += 1
            if t < T:
                state = states[-1]
                action = actions[-1]

        return total_reward, np.mean(distances), distances

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def evaluate(self, env, num_episodes=5):
        """Evaluate without exploration."""
        old_eps = self.epsilon
        self.epsilon = 0.0
        returns = []
        avg_dists = []
        for _ in range(num_episodes):
            obs, info = env.reset()
            state = obs.copy()
            total_r = 0.0
            dists = []
            for _ in range(env.max_steps):
                action = self.select_action(state)
                torque = action_to_torque(action)
                obs, r, term, trunc, info = env.step(torque)
                total_r += r
                dists.append(info["distance"])
                state = obs.copy()
                if term or trunc:
                    break
            returns.append(total_r)
            avg_dists.append(np.mean(dists))
        self.epsilon = old_eps
        return np.mean(returns), np.mean(avg_dists)


def train_sarsa(seed=42, num_episodes=100, n_step=4, target=(1.5, 0.0)):
    """Train N-step SARSA agent."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = ReacherEnv(dt=0.02, target=target)

    agent = NStepSARSA(
        state_dim=6,
        num_actions=NUM_ACTIONS,
        hidden=128,
        n_step=n_step,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.90,
    )

    episode_returns = []
    episode_errors = []

    for ep in range(num_episodes):
        total_r, avg_dist, _ = agent.train_episode(env)
        agent.decay_epsilon()
        episode_returns.append(total_r)
        episode_errors.append(avg_dist)

        if (ep + 1) % 10 == 0:
            eval_return, eval_dist = agent.evaluate(env, num_episodes=3)
            print(f"Episode {ep+1}/{num_episodes} | "
                  f"Train Return: {total_r:.1f} | Train Dist: {avg_dist:.3f} | "
                  f"Eval Return: {eval_return:.1f} | Eval Dist: {eval_dist:.3f} | "
                  f"ε: {agent.epsilon:.3f}")

    env.close()
    return agent, episode_returns, episode_errors


def main():
    print("=" * 60)
    print("Part (d): N-Step SARSA with Neural Network Q-Function")
    print("=" * 60)

    seeds = [42, 123, 456]
    all_returns = []
    all_errors = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        agent, returns, errors = train_sarsa(seed=seed, num_episodes=100, n_step=4)
        all_returns.append(returns)
        all_errors.append(errors)

    # Convert to arrays
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
    axes[0].set_title("N-Step SARSA: Episode Return")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].grid(True)

    axes[1].plot(episodes, mean_errors, 'r-')
    axes[1].fill_between(episodes, mean_errors - std_errors, mean_errors + std_errors,
                          alpha=0.3)
    axes[1].set_title("N-Step SARSA: Average Tracking Error")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Mean Distance")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "sarsa_learning_curves.pdf"))
    plt.close()
    print("\nPlot saved to sarsa_learning_curves.pdf")

    # Final evaluation
    print(f"\nFinal performance (last 10 episodes, mean±std across seeds):")
    print(f"  Return: {np.mean(all_returns[:, -10:]):.2f} ± {np.std(all_returns[:, -10:]):.2f}")
    print(f"  Error:  {np.mean(all_errors[:, -10:]):.4f} ± {np.std(all_errors[:, -10:]):.4f}")


if __name__ == "__main__":
    main()
