"""
Part (g): Generalization Tests (Target Distribution Shift)
============================================================
Train DDPG on a training distribution (circular motion), then evaluate on shifted distributions:
  - Different ω
  - Different radius r and/or center (xc, yc)
  - Lissajous trajectory
"""
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from part_e_ddpg import DDPGAgent, ReacherEnv
from part_b_pid_cartesian import CircularTargetEnv

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "Heriot_Watt_Thesis_Template-main", "Figures")
os.makedirs(IMAGES_DIR, exist_ok=True)


class LissajousTargetEnv(ReacherEnv):
    """ReacherEnv with Lissajous curve as target trajectory."""
    def __init__(self, scale=0.8, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.time = 0.0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.time = 0.0
        self._update_target()
        obs = self._get_obs()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.time += self.dt
        self._update_target()
        p_end = self._forward_kinematics(self.state[:2])
        loss = np.sum((p_end - self.target) ** 2)
        reward = -loss
        info["target"] = self.target.copy()
        info["loss"] = loss
        info["distance"] = np.sqrt(loss)
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _update_target(self):
        t = self.time
        self.target = np.array([
            self.scale * np.sin(2 * t + np.pi / 2),
            self.scale * np.sin(4 * t)
        ])


def train_ddpg_on_env(env_class, env_kwargs, seed, num_episodes=1000):
    """Train DDPG on specified environment."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = env_class(**env_kwargs)
    agent = DDPGAgent(
        state_dim=8, action_dim=2, hidden=256, hidden2=128,
        actor_lr=2e-4, critic_lr=6e-4, gamma=0.99, tau=0.005,
        batch_size=256, buffer_size=200000,
        noise_type="ou", noise_sigma=0.4,
        reward_scale=1, grad_clip=1.0, warmup_steps=3000,
        updates_per_step=1, noise_decay=0.998,
    )

    for ep in range(num_episodes):
        agent.train_episode(env)

    env.close()
    return agent


def evaluate_agent_on_env(agent, env_class, env_kwargs, num_episodes=5, eps=0.2):
    """Evaluate trained agent on a test environment."""
    env = env_class(**env_kwargs)
    all_errors = []
    all_success = []
    all_returns = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        state = agent.get_extended_state(obs, env)
        total_r = 0.0
        dists = []

        for _ in range(env.max_steps):
            action = agent.select_action(state, add_noise=False)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            dists.append(info["distance"])
            state = agent.get_extended_state(obs, env)
            if term or trunc:
                break

        all_errors.append(np.mean(dists))
        all_success.append(np.mean([1.0 if d < eps else 0.0 for d in dists]))
        all_returns.append(total_r)

    env.close()
    return np.mean(all_errors), np.mean(all_success), np.mean(all_returns)


def main():
    print("=" * 60)
    print("Part (g): Generalization Tests")
    print("=" * 60)

    seeds = [42, 123, 456]
    num_episodes = 1000

    # Training distribution: circular motion with default params
    omega_train = 2 * np.pi / 3 * 0.2
    train_env_class = CircularTargetEnv
    train_env_kwargs = {"xc": 0.0, "yc": 0.0, "r": 0.5, "omega": omega_train, "dt": 0.02}

    # Test distributions
    test_envs = {
        "Train (circular ω=0.42)": (CircularTargetEnv,
            {"xc": 0.0, "yc": 0.0, "r": 0.5, "omega": omega_train, "dt": 0.02}),
        "Diff ω (ω=1.0)": (CircularTargetEnv,
            {"xc": 0.0, "yc": 0.0, "r": 0.5, "omega": 1.0, "dt": 0.02}),
        "Diff ω (ω=2.0)": (CircularTargetEnv,
            {"xc": 0.0, "yc": 0.0, "r": 0.5, "omega": 2.0, "dt": 0.02}),
        "Diff r=1.0": (CircularTargetEnv,
            {"xc": 0.0, "yc": 0.0, "r": 1.0, "omega": omega_train, "dt": 0.02}),
        "Diff center (0.5,0.5)": (CircularTargetEnv,
            {"xc": 0.5, "yc": 0.5, "r": 0.5, "omega": omega_train, "dt": 0.02}),
        "Lissajous": (LissajousTargetEnv,
            {"scale": 0.8, "dt": 0.02}),
        "Static (1.5, 0)": (ReacherEnv,
            {"target": (1.5, 0.0), "dt": 0.02}),
    }

    # Results
    all_results = {name: {"errors": [], "success": [], "returns": []}
                   for name in test_envs}

    for seed in seeds:
        print(f"\n--- Training with seed {seed} ---")
        agent = train_ddpg_on_env(train_env_class, train_env_kwargs, seed, num_episodes)

        for name, (env_class, env_kwargs) in test_envs.items():
            avg_err, avg_suc, avg_ret = evaluate_agent_on_env(agent, env_class, env_kwargs)
            all_results[name]["errors"].append(avg_err)
            all_results[name]["success"].append(avg_suc)
            all_results[name]["returns"].append(avg_ret)
            print(f"  {name}: error={avg_err:.3f}, success={avg_suc:.3f}")

    # --- Results table ---
    print("\n" + "=" * 70)
    print(f"{'Distribution':<28} {'Avg Error':<16} {'Success Rate':<16} {'Return'}")
    print("-" * 70)
    for name in test_envs:
        errs = all_results[name]["errors"]
        succ = all_results[name]["success"]
        rets = all_results[name]["returns"]
        print(f"{name:<28} {np.mean(errs):.4f}±{np.std(errs):.4f}  "
              f"{np.mean(succ):.4f}±{np.std(succ):.4f}  "
              f"{np.mean(rets):.1f}±{np.std(rets):.1f}")

    # --- Bar chart ---
    names = list(test_envs.keys())
    errors_mean = [np.mean(all_results[n]["errors"]) for n in names]
    errors_std = [np.std(all_results[n]["errors"]) for n in names]
    success_mean = [np.mean(all_results[n]["success"]) for n in names]
    success_std = [np.std(all_results[n]["success"]) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(names))

    ax1.bar(x, errors_mean, yerr=errors_std, capsize=3, color="coral", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("Average Tracking Error")
    ax1.set_title("Generalization: Average Error")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, success_mean, yerr=success_std, capsize=3, color="skyblue", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Success Rate (ε=0.2)")
    ax2.set_title("Generalization: Success Rate")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "generalization_tests.pdf"))
    plt.close()
    print("\nPlot saved to generalization_tests.pdf")


if __name__ == "__main__":
    main()
