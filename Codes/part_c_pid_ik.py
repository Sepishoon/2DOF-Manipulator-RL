"""
Part (c): Joint-Space PID + Inverse Kinematics
================================================
Solve IK to get desired joint angles, then track with joint-space PID.
Two scenarios: static target and circular motion target.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from reacher_env import ReacherEnv
from part_b_pid_cartesian import CircularTargetEnv, save_video

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "Heriot_Watt_Thesis_Template-main", "Figures")
os.makedirs(IMAGES_DIR, exist_ok=True)


class InverseKinematics:
    """Analytical IK for 2-link planar arm (elbow-up solution)."""

    def __init__(self, l1=1.0, l2=1.0):
        self.l1 = l1
        self.l2 = l2

    def solve(self, xt, yt):
        """Return (θ1_d, θ2_d) for target (xt, yt). Elbow-up."""
        l1, l2 = self.l1, self.l2
        r2 = xt**2 + yt**2
        r = np.sqrt(r2)

        # Clamp to reachable workspace
        max_reach = l1 + l2 - 1e-4
        min_reach = abs(l1 - l2) + 1e-4
        if r > max_reach:
            xt = xt / r * max_reach
            yt = yt / r * max_reach
            r2 = xt**2 + yt**2
        elif r < min_reach:
            xt = xt / r * min_reach
            yt = yt / r * min_reach
            r2 = xt**2 + yt**2

        cos_theta2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        theta2 = np.arccos(cos_theta2)  # elbow-up (positive θ2)

        alpha = np.arctan2(yt, xt)
        beta = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
        theta1 = alpha - beta

        return theta1, theta2


class JointSpacePID:
    """Two independent PID controllers for joint-space tracking."""

    def __init__(self, Kp1=20.0, Ki1=2.0, Kd1=5.0, Kp2=20.0, Ki2=2.0, Kd2=5.0):
        self.Kp = np.array([Kp1, Kp2])
        self.Ki = np.array([Ki1, Ki2])
        self.Kd = np.array([Kd1, Kd2])
        self.integral = np.zeros(2)
        self.prev_error = None

    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = None

    def compute(self, theta_d, theta, dtheta, dt):
        """
        theta_d: desired joint angles (2,)
        theta: current joint angles (2,)
        dtheta: current joint velocities (2,)
        """
        error = theta_d - theta
        # Wrap error to [-π, π]
        error = ((error + np.pi) % (2 * np.pi)) - np.pi

        self.integral += error * dt

        if self.prev_error is None:
            derivative = np.zeros(2)
        else:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error.copy()

        tau = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return tau


class IKPIDController:
    """Combined IK + Joint-Space PID controller."""

    def __init__(self, Kp1=20.0, Ki1=2.0, Kd1=5.0, Kp2=20.0, Ki2=2.0, Kd2=5.0):
        self.ik = InverseKinematics()
        self.pid = JointSpacePID(Kp1, Ki1, Kd1, Kp2, Ki2, Kd2)

    def reset(self):
        self.pid.reset()

    def compute(self, obs, dt):
        """obs = (x_t, y_t, θ1, θ2, dθ1, dθ2)"""
        x_t, y_t, θ1, θ2, dθ1, dθ2 = obs

        # Solve IK
        θ1_d, θ2_d = self.ik.solve(x_t, y_t)
        theta_d = np.array([θ1_d, θ2_d])
        theta = np.array([θ1, θ2])
        dtheta = np.array([dθ1, dθ2])

        # Joint-space PID
        tau = self.pid.compute(theta_d, theta, dtheta, dt)
        return tau


def run_pid_scenario(env, controller, scenario_name, dt=0.02):
    """Run one episode and collect data."""
    obs, info = env.reset(seed=42)
    controller.reset()

    frames = []
    if env.render_mode == "rgb_array":
        frames.append(env.render())

    distances = []
    total_reward = 0.0

    for step in range(env.max_steps):
        tau = controller.compute(obs, dt)
        obs, reward, terminated, truncated, info = env.step(tau)
        total_reward += reward
        distances.append(info["distance"])

        if env.render_mode == "rgb_array":
            frames.append(env.render())

    avg_error = np.mean(distances)
    print(f"[{scenario_name}] Avg tracking error: {avg_error:.4f}, "
          f"Final distance: {distances[-1]:.4f}, Total reward: {total_reward:.2f}")
    return frames, distances, avg_error


def main():
    dt = 0.02
    # Tuned gains
    Kp1, Ki1, Kd1 = 20.0, 0.0, 10.0
    Kp2, Ki2, Kd2 = 20.0, 0.0, 10.0
    print(f"Joint PID Gains: Kp={Kp1}, Ki={Ki1}, Kd={Kd1}")

    # --- Scenario 1: Static target ---
    print("\n=== Scenario 1: Static Target (1.5, 0) ===")
    env_static = ReacherEnv(dt=dt, target=(1.5, 0.0), render_mode="rgb_array")
    controller = IKPIDController(Kp1, Ki1, Kd1, Kp2, Ki2, Kd2)
    frames_s, dist_s, avg_s = run_pid_scenario(env_static, controller, "Static")
    save_video(frames_s, os.path.join(IMAGES_DIR, "pid_ik_static.mp4"))
    env_static.close()

    # --- Scenario 2: Circular motion ---
    print("\n=== Scenario 2: Circular Motion ===")
    omega = 2 * np.pi / 3 * 0.2
    env_circ = CircularTargetEnv(
        xc=0.0, yc=0.0, r=0.5, omega=omega,
        dt=dt, render_mode="rgb_array"
    )
    controller2 = IKPIDController(Kp1, Ki1, Kd1, Kp2, Ki2, Kd2)
    frames_c, dist_c, avg_c = run_pid_scenario(env_circ, controller2, "Circular")
    save_video(frames_c, os.path.join(IMAGES_DIR, "pid_ik_circular.mp4"))
    env_circ.close()

    # --- Plot tracking errors ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(dist_s)
    axes[0].set_title(f"IK+PID Static Target - Avg Error: {avg_s:.4f}")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Distance to Target")
    axes[0].grid(True)

    axes[1].plot(dist_c)
    axes[1].set_title(f"IK+PID Circular Target - Avg Error: {avg_c:.4f}")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Distance to Target")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "pid_ik_errors.pdf"))
    plt.close()
    print("\nPlot saved to pid_ik_errors.pdf")


if __name__ == "__main__":
    main()
