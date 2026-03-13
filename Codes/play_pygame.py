"""
Interactive Pygame application for the Two-Link Reacher.
Controls:
  W/S      : increase/decrease torque on joint 1
  UP/DOWN  : increase/decrease torque on joint 2
  +/-      : increase/decrease torque magnitude
  R        : reset environment
  Q / ESC  : quit

The environment runs continuously (auto-resets on truncation).
"""
import numpy as np
import pygame
from reacher_env import ReacherEnv


def main():
    env = ReacherEnv(dt=0.02, target=(1.5, 0.0), render_mode="human",
                     max_steps=10000)
    obs, info = env.reset(seed=0)
    env.render()

    tau = np.array([0.0, 0.0])
    torque_mag = 0.3
    fps = int(1.0 / env.dt)

    clock = pygame.time.Clock()
    running = True
    step_count = 0

    print("Controls:")
    print("  W/S      = joint 1 torque (+/-)")
    print("  UP/DOWN  = joint 2 torque (+/-)")
    print("  +/=  -   = increase/decrease torque magnitude")
    print("  R        = reset environment")
    print("  Q / ESC  = quit")
    print(f"Torque magnitude: {torque_mag:.2f} | Running at ~{fps} FPS")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    tau = np.array([0.0, 0.0])
                    step_count = 0
                    print("Environment reset!")
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS,
                                   pygame.K_KP_PLUS):
                    torque_mag = min(1.0, torque_mag + 0.1)
                    print(f"Torque magnitude: {torque_mag:.2f}")
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    torque_mag = max(0.1, torque_mag - 0.1)
                    print(f"Torque magnitude: {torque_mag:.2f}")

        # Continuous key press
        keys = pygame.key.get_pressed()
        tau[0] = 0.0
        tau[1] = 0.0
        if keys[pygame.K_w]:
            tau[0] += torque_mag
        if keys[pygame.K_s]:
            tau[0] -= torque_mag
        if keys[pygame.K_UP]:
            tau[1] += torque_mag
        if keys[pygame.K_DOWN]:
            tau[1] -= torque_mag

        obs, reward, terminated, truncated, info = env.step(tau)
        env.render()
        step_count += 1

        if truncated or terminated:
            obs, info = env.reset()
            tau = np.array([0.0, 0.0])
            step_count = 0

        clock.tick(fps)

    env.close()


if __name__ == "__main__":
    main()
