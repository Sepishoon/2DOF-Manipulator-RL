"""
Test script for Part (a): run one episode with constant torques τ1=τ2=1
and save a video using matplotlib animation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from reacher_env import ReacherEnv


def run_episode_and_record():
    env = ReacherEnv(dt=0.02, target=(1.5, 0.0), render_mode="rgb_array")
    obs, info = env.reset(seed=0)

    frames = []
    frames.append(env.render())

    total_reward = 0.0
    for step in range(150):
        action = np.array([1.0, 1.0])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frames.append(env.render())

    env.close()

    print(f"Episode finished: total_reward={total_reward:.2f}, "
          f"final_distance={info['distance']:.4f}")

    # Save video using matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=20, blit=True)
    ani.save("test_constant_torque.mp4", writer="ffmpeg", fps=50)
    print("Video saved to test_constant_torque.mp4")
    plt.close()


if __name__ == "__main__":
    run_episode_and_record()
