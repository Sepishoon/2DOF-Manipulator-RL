"""
Part (i): Metrics and Reporting Standards
==========================================
For every method (PID, IK+PID, SARSA, DDPG), report:
  1. Mean tracking error
  2. Success rate (ε=0.2)
  3. Control energy
  4. Smoothness
Run RL methods with K=3 seeds, plot mean ± std.
"""
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from reacher_env import ReacherEnv
from part_b_pid_cartesian import CartesianPID
from part_c_pid_ik import IKPIDController
from part_d_sarsa import NStepSARSA, action_to_torque, NUM_ACTIONS
from part_e_ddpg import DDPGAgent

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "Heriot_Watt_Thesis_Template-main", "Figures")
os.makedirs(IMAGES_DIR, exist_ok=True)


EPS = 0.2  # success threshold


def compute_metrics(distances, torques):
    """Compute all standardized metrics."""
    mean_error = np.mean(distances)
    success_rate = np.mean([1.0 if d < EPS else 0.0 for d in distances])
    control_energy = sum(np.sum(t**2) for t in torques)
    smoothness = 0.0
    for i in range(1, len(torques)):
        smoothness += np.sum((torques[i] - torques[i-1])**2)
    return {
        "mean_error": mean_error,
        "success_rate": success_rate,
        "control_energy": control_energy,
        "smoothness": smoothness,
    }


def evaluate_pid_cartesian(target=(1.5, 0.0), seed=42):
    env = ReacherEnv(dt=0.02, target=target)
    obs, info = env.reset(seed=seed+1000)
    controller = CartesianPID(Kp=10.0, Ki=0.0, Kd=8.0)
    controller.reset()
    distances, torques = [], []
    for _ in range(env.max_steps):
        tau = controller.compute(obs, env.dt)
        tau_clipped = np.clip(tau, -1.0, 1.0)
        torques.append(tau_clipped.copy())
        obs, reward, terminated, truncated, info = env.step(tau)
        distances.append(info["distance"])
    env.close()
    return compute_metrics(distances, torques)


def evaluate_ik_pid(target=(1.5, 0.0), seed=42):
    env = ReacherEnv(dt=0.02, target=target)
    obs, info = env.reset(seed=seed+1000)
    controller = IKPIDController(20.0, 0.0, 10.0, 20.0, 0.0, 10.0)
    controller.reset()
    distances, torques = [], []
    for _ in range(env.max_steps):
        tau = controller.compute(obs, env.dt)
        tau_clipped = np.clip(tau, -1.0, 1.0)
        torques.append(tau_clipped.copy())
        obs, reward, terminated, truncated, info = env.step(tau)
        distances.append(info["distance"])
    env.close()
    return compute_metrics(distances, torques)


def evaluate_sarsa(target=(1.5, 0.0), seed=42, n_step=4):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = ReacherEnv(dt=0.02, target=target)
    agent = NStepSARSA(
        state_dim=6, num_actions=NUM_ACTIONS, hidden=128,
        n_step=n_step, lr=1e-3, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.90,
    )
    for _ in range(100):
        agent.train_episode(env)
        agent.decay_epsilon()
    agent.epsilon = 0.0
    obs, info = env.reset(seed=seed+1000)
    state = obs.copy()
    distances, torques = [], []
    for _ in range(env.max_steps):
        action = agent.select_action(state)
        tau = action_to_torque(action)
        torques.append(tau.copy())
        obs, r, term, trunc, info = env.step(tau)
        distances.append(info["distance"])
        state = obs.copy()
        if term or trunc:
            break
    env.close()
    return compute_metrics(distances, torques)


def evaluate_ddpg(target=(1.5, 0.0), seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env = ReacherEnv(dt=0.02, target=target)
    agent = DDPGAgent(
        state_dim=8, action_dim=2, hidden=64, hidden2=32,
        actor_lr=2e-3, critic_lr=4e-3, gamma=0.99, tau=0.005,
        batch_size=64, buffer_size=50000,
        noise_type="ou", noise_sigma=0.4,
        reward_scale=0.5, grad_clip=1.0, warmup_steps=32,
        updates_per_step=1, noise_decay=0.93,
    )
    for _ in range(100):
        agent.train_episode(env)
    # Evaluate on same seed as other methods
    obs, info = env.reset(seed=seed+1000)
    state = agent.get_extended_state(obs, env)
    distances, torques = [], []
    for _ in range(env.max_steps):
        action = agent.select_action(state, add_noise=False)
        torques.append(action.copy())
        obs, r, term, trunc, info = env.step(action)
        distances.append(info["distance"])
        state = agent.get_extended_state(obs, env)
        if term or trunc:
            break
    env.close()
    return compute_metrics(distances, torques)


def main():
    print("=" * 70)
    print("Part (i): Metrics and Reporting Standards")
    print("=" * 70)

    seeds = [42, 123, 456]
    target = (1.5, 0.0)

    methods = {
        "PID (Cartesian)": evaluate_pid_cartesian,
        "IK + PID (Joint)": evaluate_ik_pid,
        "N-Step SARSA": evaluate_sarsa,
        "DDPG": evaluate_ddpg,
    }

    all_metrics = {name: {k: [] for k in ["mean_error", "success_rate",
                                           "control_energy", "smoothness"]}
                   for name in methods}

    for name, eval_fn in methods.items():
        print(f"\n--- {name} ---")
        for seed in seeds:
            metrics = eval_fn(target=target, seed=seed)
            for k, v in metrics.items():
                all_metrics[name][k].append(v)
            print(f"  Seed {seed}: error={metrics['mean_error']:.4f}, "
                  f"success={metrics['success_rate']:.4f}")

    # --- Final Comparison Table ---
    print("\n" + "=" * 90)
    print(f"{'Method':<20} {'Mean Error':<18} {'Success Rate':<18} "
          f"{'Ctrl Energy':<18} {'Smoothness'}")
    print("-" * 90)

    table_data = {}
    for name in methods:
        m = all_metrics[name]
        row = {}
        for k in ["mean_error", "success_rate", "control_energy", "smoothness"]:
            mean_v = np.mean(m[k])
            std_v = np.std(m[k])
            row[k] = (mean_v, std_v)
        table_data[name] = row
        print(f"{name:<20} "
              f"{row['mean_error'][0]:.4f}±{row['mean_error'][1]:.4f}  "
              f"{row['success_rate'][0]:.4f}±{row['success_rate'][1]:.4f}  "
              f"{row['control_energy'][0]:.1f}±{row['control_energy'][1]:.1f}      "
              f"{row['smoothness'][0]:.1f}±{row['smoothness'][1]:.1f}")

    # --- Visualization ---
    metric_names = ["mean_error", "success_rate", "control_energy", "smoothness"]
    metric_labels = ["Mean Tracking Error", "Success Rate (ε=0.2)",
                     "Control Energy", "Smoothness (Δτ²)"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    method_names = list(methods.keys())
    x = np.arange(len(method_names))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx // 2][idx % 2]
        means = [table_data[n][metric][0] for n in method_names]
        stds = [table_data[n][metric][1] for n in method_names]
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=15, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "final_metrics_comparison.pdf"))
    plt.close()
    print("\nPlot saved to final_metrics_comparison.pdf")


if __name__ == "__main__":
    main()
