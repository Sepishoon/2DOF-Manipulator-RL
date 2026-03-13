"""
Part (h): Ablation Study on State Inputs
==========================================
Train DDPG with 4 different state representations:
  1. Minimal:            s = (x_t, y_t, θ1, θ2, dθ1, dθ2)
  2. Minimal+end:        s = (x_t, y_t, θ1, θ2, dθ1, dθ2, x_end, y_end)
  3. Error-only:         s = (x_t - x_end, y_t - y_end, θ1, θ2, dθ1, dθ2)
  4. Error in IK:        s = (θ1 - θ1_target, θ2 - θ2_target, dθ1, dθ2)
"""
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from part_e_ddpg import DDPGAgent, Actor, Critic, ReplayBuffer, OUNoise, ReacherEnv
from part_c_pid_ik import InverseKinematics

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "Heriot_Watt_Thesis_Template-main", "Figures")
os.makedirs(IMAGES_DIR, exist_ok=True)


class DDPGAgentCustomState(DDPGAgent):
    """DDPG agent with custom state representation."""

    def __init__(self, state_type="minimal+end", **kwargs):
        self.state_type = state_type
        self.ik = InverseKinematics()
        if state_type == "minimal":
            kwargs["state_dim"] = 6
        elif state_type == "minimal+end":
            kwargs["state_dim"] = 8
        elif state_type == "error-only":
            kwargs["state_dim"] = 6
        elif state_type == "error-ik":
            kwargs["state_dim"] = 4
        super().__init__(**kwargs)

    def get_extended_state(self, obs, env):
        x_t, y_t, θ1, θ2, dθ1, dθ2 = obs
        x_end = env.l1 * np.cos(θ1) + env.l2 * np.cos(θ1 + θ2)
        y_end = env.l1 * np.sin(θ1) + env.l2 * np.sin(θ1 + θ2)

        if self.state_type == "minimal":
            return np.array([x_t, y_t, θ1, θ2, dθ1, dθ2], dtype=np.float32)
        elif self.state_type == "minimal+end":
            return np.array([x_t, y_t, θ1, θ2, dθ1, dθ2, x_end, y_end], dtype=np.float32)
        elif self.state_type == "error-only":
            return np.array([x_t - x_end, y_t - y_end, θ1, θ2, dθ1, dθ2], dtype=np.float32)
        elif self.state_type == "error-ik":
            θ1_t, θ2_t = self.ik.solve(x_t, y_t)
            e1 = ((θ1 - θ1_t + np.pi) % (2 * np.pi)) - np.pi
            e2 = ((θ2 - θ2_t + np.pi) % (2 * np.pi)) - np.pi
            return np.array([e1, e2, dθ1, dθ2], dtype=np.float32)
        else:
            raise ValueError(f"Unknown state_type: {self.state_type}")


def train_and_evaluate(state_type, seed, num_episodes=1000, target=(1.5, 0.0)):
    """Train and return learning curve for a specific state representation."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = ReacherEnv(dt=0.02, target=target)

    agent = DDPGAgentCustomState(
        state_type=state_type,
        action_dim=2, hidden=256, hidden2=128,
        actor_lr=2e-4, critic_lr=6e-4, gamma=0.99, tau=0.005,
        batch_size=256, buffer_size=200000,
        noise_type="ou", noise_sigma=0.4,
        reward_scale=1, grad_clip=1.0, warmup_steps=3000,
        updates_per_step=1, noise_decay=0.998,
    )

    episode_returns = []
    episode_errors = []

    for ep in range(num_episodes):
        total_r, avg_dist, _ = agent.train_episode(env)
        episode_returns.append(total_r)
        episode_errors.append(avg_dist)

    env.close()
    return episode_returns, episode_errors


def main():
    print("=" * 60)
    print("Part (h): Ablation Study on State Inputs")
    print("=" * 60)

    num_episodes = 1000
    seeds = [42, 123, 456]

    state_types = {
        "Minimal (6-dim)": "minimal",
        "Minimal+End (8-dim)": "minimal+end",
        "Error-Only (6-dim)": "error-only",
        "Error-IK (4-dim)": "error-ik",
    }

    results = {}

    for label, state_type in state_types.items():
        print(f"\n--- {label} ---")
        all_returns = []
        all_errors = []

        for seed in seeds:
            returns, errors = train_and_evaluate(state_type, seed, num_episodes)
            all_returns.append(returns)
            all_errors.append(errors)
            print(f"  Seed {seed}: Final Avg Error = {np.mean(errors[-10:]):.3f}")

        results[label] = {
            "returns": np.array(all_returns),
            "errors": np.array(all_errors),
        }

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    episodes = np.arange(1, num_episodes + 1)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, (label, data) in enumerate(results.items()):
        color = colors[idx]

        mean_r = np.mean(data["returns"], axis=0)
        std_r = np.std(data["returns"], axis=0)
        axes[0].plot(episodes, mean_r, color=color, label=label)
        axes[0].fill_between(episodes, mean_r - std_r, mean_r + std_r, alpha=0.15, color=color)

        mean_e = np.mean(data["errors"], axis=0)
        std_e = np.std(data["errors"], axis=0)
        axes[1].plot(episodes, mean_e, color=color, label=label)
        axes[1].fill_between(episodes, mean_e - std_e, mean_e + std_e, alpha=0.15, color=color)

    axes[0].set_title("Episode Return by State Representation")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].legend(fontsize=9)
    axes[0].grid(True)

    axes[1].set_title("Average Tracking Error by State Representation")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Mean Distance")
    axes[1].legend(fontsize=9)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "ablation_state_inputs.pdf"))
    plt.close()
    print("\nPlot saved to ablation_state_inputs.pdf")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Representation':<25} {'Avg Error (last 10)':<22} {'Avg Return (last 10)'}")
    print("-" * 60)
    for label in results:
        avg_err = np.mean(results[label]["errors"][:, -10:])
        std_err = np.std(results[label]["errors"][:, -10:])
        avg_ret = np.mean(results[label]["returns"][:, -10:])
        std_ret = np.std(results[label]["returns"][:, -10:])
        print(f"{label:<25} {avg_err:.4f}±{std_err:.4f}      {avg_ret:.1f}±{std_ret:.1f}")


if __name__ == "__main__":
    main()
