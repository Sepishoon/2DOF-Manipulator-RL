"""
Part (f): Exploration and Action Noise Study (DDPG-Specific)
=============================================================
Compare OU noise, Gaussian noise, and No noise using the top1 DDPG configuration.
Plot learning curves (mean ± std over K=3 seeds) and success rate.
"""
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from part_e_ddpg import DDPGAgent, ReacherEnv

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "Heriot_Watt_Thesis_Template-main", "Figures")
os.makedirs(IMAGES_DIR, exist_ok=True)


class NoNoise:
    """No exploration noise."""
    def __init__(self, size):
        self.size = size

    def reset(self):
        pass

    def sample(self):
        return np.zeros(self.size)

    def decay(self, factor):
        pass


def train_ddpg_with_config(seed, num_episodes, noise_type, noise_sigma, target=(1.5, 0.0)):
    """Train DDPG with specific noise config and return per-episode metrics."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = ReacherEnv(dt=0.02, target=target)

    # Use part_e_ddpg hyperparameters for all configs
    actual_noise_type = noise_type if noise_type != "none" else "ou"
    agent = DDPGAgent(
        state_dim=8, action_dim=2, hidden=256, hidden2=128,
        actor_lr=2e-4, critic_lr=6e-4, gamma=0.99, tau=0.005,
        batch_size=256, buffer_size=200000,
        noise_type=actual_noise_type, noise_sigma=noise_sigma,
        reward_scale=1, grad_clip=1.0, warmup_steps=3000,
        updates_per_step=1, noise_decay=0.998,
    )

    # Replace noise with NoNoise if noise_type is "none"
    if noise_type == "none":
        agent.noise = NoNoise(2)

    episode_returns = []
    episode_errors = []
    episode_success_rates = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        state = agent.get_extended_state(obs, env)
        agent.noise.reset()

        total_reward = 0.0
        distances = []

        for step in range(env.max_steps):
            action = agent.select_action(state, add_noise=True)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = agent.get_extended_state(obs, env)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.total_steps += 1
            total_reward += reward
            distances.append(info["distance"])

            agent.update()
            state = next_state
            if done:
                break

        # Decay noise
        if agent.noise_decay < 1.0:
            agent.noise.decay(agent.noise_decay)

        episode_returns.append(total_reward)
        episode_errors.append(np.mean(distances))

        # Success rate: fraction of steps where distance < epsilon
        eps = 0.2
        success_rate = np.mean([1.0 if d < eps else 0.0 for d in distances])
        episode_success_rates.append(success_rate)

    env.close()
    return episode_returns, episode_errors, episode_success_rates


def main():
    print("=" * 60)
    print("Part (f): Exploration and Action Noise Study")
    print("=" * 60)

    num_episodes = 1000
    seeds = [42, 123, 456]

    # Three noise configurations using part_e_ddpg sigma=0.4
    configs = [
        ("OU Noise (σ=0.4)", "ou", 0.4),
        ("Gaussian Noise (σ=0.4)", "gaussian", 0.4),
        ("No Noise", "none", 0.0),
    ]

    results = {}

    for name, noise_type, sigma in configs:
        print(f"\n--- {name} ---")
        all_returns = []
        all_errors = []
        all_success = []

        for seed in seeds:
            returns, errors, success = train_ddpg_with_config(
                seed, num_episodes, noise_type, sigma
            )
            all_returns.append(returns)
            all_errors.append(errors)
            all_success.append(success)
            print(f"  Seed {seed}: Final Avg Error = {np.mean(errors[-10:]):.3f}")

        results[name] = {
            "returns": np.array(all_returns),
            "errors": np.array(all_errors),
            "success": np.array(all_success),
        }

    # --- Plot learning curves ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    episodes = np.arange(1, num_episodes + 1)
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for idx, (name, data) in enumerate(results.items()):
        color = colors[idx]

        # Returns
        mean_r = np.mean(data["returns"], axis=0)
        std_r = np.std(data["returns"], axis=0)
        axes[0, 0].plot(episodes, mean_r, color=color, label=name)
        axes[0, 0].fill_between(episodes, mean_r - std_r, mean_r + std_r, alpha=0.15, color=color)

        # Errors
        mean_e = np.mean(data["errors"], axis=0)
        std_e = np.std(data["errors"], axis=0)
        axes[0, 1].plot(episodes, mean_e, color=color, label=name)
        axes[0, 1].fill_between(episodes, mean_e - std_e, mean_e + std_e, alpha=0.15, color=color)

        # Success rate
        mean_s = np.mean(data["success"], axis=0)
        std_s = np.std(data["success"], axis=0)
        axes[1, 0].plot(episodes, mean_s, color=color, label=name)
        axes[1, 0].fill_between(episodes, mean_s - std_s, mean_s + std_s, alpha=0.15, color=color)

    axes[0, 0].set_title("Episode Return")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True)

    axes[0, 1].set_title("Average Tracking Error")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Mean Distance")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True)

    axes[1, 0].set_title("Success Rate (ε=0.2)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Success Rate")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True)

    # Bar chart of final performance
    names = list(results.keys())
    final_errors = [np.mean(results[n]["errors"][:, -10:]) for n in names]
    final_success = [np.mean(results[n]["success"][:, -10:]) for n in names]

    x = np.arange(len(names))
    axes[1, 1].bar(x - 0.2, final_errors, 0.4, label="Avg Error (last 10 ep)", color="salmon")
    ax2 = axes[1, 1].twinx()
    ax2.bar(x + 0.2, final_success, 0.4, label="Success Rate (last 10 ep)", color="skyblue")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    axes[1, 1].set_ylabel("Avg Error", color="salmon")
    ax2.set_ylabel("Success Rate", color="skyblue")
    axes[1, 1].set_title("Final Performance Comparison")
    axes[1, 1].legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "noise_study.pdf"))
    plt.close()
    print("\nPlot saved to noise_study.pdf")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"{'Config':<28} {'Avg Error (last 10)':<22} {'Success Rate (last 10)'}")
    print("-" * 60)
    for name in names:
        avg_err = np.mean(results[name]["errors"][:, -10:])
        avg_suc = np.mean(results[name]["success"][:, -10:])
        print(f"{name:<28} {avg_err:<22.4f} {avg_suc:.4f}")


if __name__ == "__main__":
    main()
