"""
Part (b): Cartesian-Space PID Control
======================================
PID controller in task space using Jacobian transpose to map forces to torques.
Two scenarios: static target and circular motion target.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from reacher_env import ReacherEnv

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "Heriot_Watt_Thesis_Template-main", "Figures")
os.makedirs(IMAGES_DIR, exist_ok=True)


class CartesianPID:
    """PID controller in Cartesian (task) space with Jacobian transpose mapping."""

    def __init__(self, Kp=10.0, Ki=0.0, Kd=8.0, l1=1.0, l2=1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.l1 = l1
        self.l2 = l2
        self.integral = np.zeros(2)
        self.prev_error = None

    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = None

    def jacobian(self, θ1, θ2):
        """Compute the 2x2 Jacobian J(q)."""
        l1, l2 = self.l1, self.l2
        s1 = np.sin(θ1)
        c1 = np.cos(θ1)
        s12 = np.sin(θ1 + θ2)
        c12 = np.cos(θ1 + θ2)
        J = np.array([
            [-l1 * s1 - l2 * s12, -l2 * s12],
            [ l1 * c1 + l2 * c12,  l2 * c12]
        ])
        return J

    def compute(self, obs, dt):
        """Compute torque given observation = (x_t, y_t, θ1, θ2, dθ1, dθ2)."""
        x_t, y_t, θ1, θ2, dθ1, dθ2 = obs

        # Current end-effector position
        x_end = self.l1 * np.cos(θ1) + self.l2 * np.cos(θ1 + θ2)
        y_end = self.l1 * np.sin(θ1) + self.l2 * np.sin(θ1 + θ2)
        p_end = np.array([x_end, y_end])
        p_target = np.array([x_t, y_t])

        # Error in task space
        error = p_target - p_end

        # PID terms
        self.integral += error * dt
        if self.prev_error is None:
            derivative = np.zeros(2)
        else:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error.copy()

        # Task-space control force
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Map to joint torques via Jacobian transpose
        J = self.jacobian(θ1, θ2)
        tau = J.T @ u

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
          f"Total reward: {total_reward:.2f}")
    return frames, distances, avg_error


def save_video(frames, filename, fps=50):
    if not frames:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=True)
    ani.save(filename, writer="ffmpeg", fps=fps)
    plt.close()
    print(f"Video saved to {filename}")


class CircularTargetEnv(ReacherEnv):
    """ReacherEnv with circular moving target."""

    def __init__(self, xc=0.0, yc=0.0, r=0.5, omega=2*np.pi/3*0.2, **kwargs):
        super().__init__(**kwargs)
        self.xc = xc
        self.yc = yc
        self.r = r
        self.omega = omega
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
        # Recompute reward with updated target
        p_end = self._forward_kinematics(self.state[:2])
        loss = np.sum((p_end - self.target) ** 2)
        reward = -loss
        info["target"] = self.target.copy()
        info["loss"] = loss
        info["distance"] = np.sqrt(loss)
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _update_target(self):
        self.target = np.array([
            self.xc + self.r * np.cos(self.omega * self.time),
            self.yc + self.r * np.sin(self.omega * self.time)
        ])


def main():
    dt = 0.02
    # PD gains (Ki=0: no integral needed for 3-sec window)
    Kp, Ki, Kd = 10.0, 0.0, 8.0
    print(f"PID Gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")

    # --- Scenario 1: Static target ---
    print("\n=== Scenario 1: Static Target (1.5, 0) ===")
    env_static = ReacherEnv(dt=dt, target=(1.5, 0.0), render_mode="rgb_array")
    controller = CartesianPID(Kp=Kp, Ki=Ki, Kd=Kd)
    frames_s, dist_s, avg_s = run_pid_scenario(env_static, controller, "Static")
    save_video(frames_s, os.path.join(IMAGES_DIR, "pid_cartesian_static.mp4"))
    env_static.close()

    # --- Scenario 2: Circular motion ---
    print("\n=== Scenario 2: Circular Motion ===")
    omega = 2 * np.pi / 3 * 0.2
    env_circ = CircularTargetEnv(
        xc=0.0, yc=0.0, r=0.5, omega=omega,
        dt=dt, render_mode="rgb_array"
    )
    controller2 = CartesianPID(Kp=Kp, Ki=Ki, Kd=Kd)
    frames_c, dist_c, avg_c = run_pid_scenario(env_circ, controller2, "Circular")
    save_video(frames_c, os.path.join(IMAGES_DIR, "pid_cartesian_circular.mp4"))
    env_circ.close()

    # --- Plot tracking errors ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(dist_s)
    axes[0].set_title(f"Static Target - Avg Error: {avg_s:.4f}")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Distance to Target")
    axes[0].grid(True)

    axes[1].plot(dist_c)
    axes[1].set_title(f"Circular Target - Avg Error: {avg_c:.4f}")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Distance to Target")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "pid_cartesian_errors.pdf"))
    plt.close()
    print("\nPlot saved to pid_cartesian_errors.pdf")


if __name__ == "__main__":
    main()
