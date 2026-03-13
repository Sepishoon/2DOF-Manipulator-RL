"""
Two-Link Planar Reacher Environment (Gymnasium-style)
=====================================================
2-DOF robotic arm with torque inputs. No gravity.

Parameters:
  l1 = l2 = 1.0 m, m1 = m2 = 1.0 kg
  b1 = b2 = 0.05 kg·m/s (damping)
  dt = 0.02 s, max_steps = 150
  torque bounds: [-1, 1] N·m per joint
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ReacherEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, dt=0.02, target=None, render_mode=None, max_steps=150):
        super().__init__()
        # --- Physical parameters ---
        self.l1, self.l2 = 1.0, 1.0
        self.m1, self.m2 = 1.0, 1.0
        self.b1, self.b2 = 0.05, 0.05
        self.dt = dt
        self.max_steps = max_steps
        self.tau_max = 1.0

        # --- Target ---
        self._target_arg = target  # None ⇒ random each reset
        self.target = None

        # --- Spaces ---
        # Action: torques τ1, τ2 ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-self.tau_max, high=self.tau_max, shape=(2,), dtype=np.float64
        )
        # Observation: (x_t, y_t, θ1, θ2, θ̇1, θ̇2)
        high = np.array([3.0, 3.0, np.pi, np.pi, 20.0, 20.0], dtype=np.float64)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float64)

        # --- Render ---
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_size = 500
        self.scale = self.screen_size / 5.0  # world ∈ [-2.5, 2.5]

        # --- State ---
        self.state = None  # (θ1, θ2, θ̇1, θ̇2)
        self.steps = 0

    # -----------------------------------------------------------------
    # Forward kinematics
    # -----------------------------------------------------------------
    def _forward_kinematics(self, q):
        θ1, θ2 = q[0], q[1]
        x = self.l1 * np.cos(θ1) + self.l2 * np.cos(θ1 + θ2)
        y = self.l1 * np.sin(θ1) + self.l2 * np.sin(θ1 + θ2)
        return np.array([x, y])

    # -----------------------------------------------------------------
    # Dynamics: M(q) q̈ + C(q,q̇) q̇ + B q̇ = τ
    # -----------------------------------------------------------------
    def _dynamics(self, state, tau):
        θ1, θ2, dθ1, dθ2 = state
        l1, l2, m1, m2 = self.l1, self.l2, self.m1, self.m2

        a = m1 * l1**2 + m2 * (l1**2 + l2**2)
        b = m2 * l1 * l2
        d = m2 * l2**2

        cos2 = np.cos(θ2)
        sin2 = np.sin(θ2)

        # Inertia matrix
        M = np.array([
            [a + 2 * b * cos2, d + b * cos2],
            [d + b * cos2,     d            ]
        ])

        # Coriolis/centrifugal
        C = np.array([
            -b * sin2 * (2 * dθ1 * dθ2 + dθ2**2),
             b * sin2 * dθ1**2
        ])

        # Damping
        B_damp = np.array([self.b1 * dθ1, self.b2 * dθ2])

        # Solve M q̈ = τ - C - B  
        rhs = tau - C - B_damp
        ddq = np.linalg.solve(M, rhs)
        return ddq

    # -----------------------------------------------------------------
    # Semi-implicit Euler integration
    # -----------------------------------------------------------------
    def _step_dynamics(self, state, tau):
        ddq = self._dynamics(state, tau)
        θ1, θ2, dθ1, dθ2 = state

        # velocity update first (semi-implicit)
        dθ1_new = dθ1 + ddq[0] * self.dt
        dθ2_new = dθ2 + ddq[1] * self.dt

        # position update using new velocity
        θ1_new = θ1 + dθ1_new * self.dt
        θ2_new = θ2 + dθ2_new * self.dt

        # wrap angles to [-π, π]
        θ1_new = ((θ1_new + np.pi) % (2 * np.pi)) - np.pi
        θ2_new = ((θ2_new + np.pi) % (2 * np.pi)) - np.pi

        return np.array([θ1_new, θ2_new, dθ1_new, dθ2_new])

    # -----------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------
    def _get_obs(self):
        θ1, θ2, dθ1, dθ2 = self.state
        return np.array([
            self.target[0], self.target[1],
            θ1, θ2, dθ1, dθ2
        ], dtype=np.float64)

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random initial joint angles, zero velocities
        θ1 = self.np_random.uniform(-np.pi, np.pi)
        θ2 = self.np_random.uniform(-np.pi, np.pi)
        self.state = np.array([θ1, θ2, 0.0, 0.0])

        # Target
        if self._target_arg is not None:
            self.target = np.array(self._target_arg, dtype=np.float64)
        else:
            # Random target within reachable workspace (radius ≤ l1+l2)
            r = self.np_random.uniform(0.5, self.l1 + self.l2 - 0.1)
            angle = self.np_random.uniform(-np.pi, np.pi)
            self.target = np.array([r * np.cos(angle), r * np.sin(angle)])

        self.steps = 0
        info = {"end_effector": self._forward_kinematics(self.state[:2])}
        return self._get_obs(), info

    # -----------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -self.tau_max, self.tau_max)

        self.state = self._step_dynamics(self.state, action)
        self.steps += 1

        p_end = self._forward_kinematics(self.state[:2])
        loss = np.sum((p_end - self.target) ** 2)
        reward = -loss

        truncated = self.steps >= self.max_steps
        terminated = False

        info = {
            "end_effector": p_end,
            "target": self.target.copy(),
            "loss": loss,
            "distance": np.sqrt(loss),
        }

        return self._get_obs(), reward, terminated, truncated, info

    # -----------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None

        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for rendering. Install with: pip install pygame")

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
                pygame.display.set_caption("Two-Link Reacher")
            else:
                self.screen = pygame.Surface((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()

        # world-to-screen transform
        def w2s(wx, wy):
            sx = int(self.screen_size / 2 + wx * self.scale)
            sy = int(self.screen_size / 2 - wy * self.scale)
            return (sx, sy)

        self.screen.fill((255, 255, 255))

        θ1, θ2 = self.state[0], self.state[1]
        # Joint positions
        x0, y0 = 0.0, 0.0
        x1 = self.l1 * np.cos(θ1)
        y1 = self.l1 * np.sin(θ1)
        x2 = x1 + self.l2 * np.cos(θ1 + θ2)
        y2 = y1 + self.l2 * np.sin(θ1 + θ2)

        # Draw links
        pygame.draw.line(self.screen, (255, 180, 0), w2s(x0, y0), w2s(x1, y1), 8)
        pygame.draw.line(self.screen, (255, 100, 0), w2s(x1, y1), w2s(x2, y2), 8)

        # Draw joints
        pygame.draw.circle(self.screen, (50, 50, 200), w2s(x0, y0), 10)
        pygame.draw.circle(self.screen, (50, 50, 200), w2s(x1, y1), 8)

        # Draw end-effector
        pygame.draw.circle(self.screen, (0, 200, 0), w2s(x2, y2), 7)

        # Draw target
        tx, ty = self.target
        pygame.draw.circle(self.screen, (200, 0, 0), w2s(tx, ty), 8)
        # cross on target
        tc = w2s(tx, ty)
        sz = 10
        pygame.draw.line(self.screen, (200, 0, 0), (tc[0]-sz, tc[1]-sz), (tc[0]+sz, tc[1]+sz), 3)
        pygame.draw.line(self.screen, (200, 0, 0), (tc[0]-sz, tc[1]+sz), (tc[0]+sz, tc[1]-sz), 3)

        # Draw workspace boundary circle
        pygame.draw.circle(self.screen, (220, 220, 220), w2s(0, 0),
                           int((self.l1 + self.l2) * self.scale), 1)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            ).copy()

    # -----------------------------------------------------------------
    # Close
    # -----------------------------------------------------------------
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
            self.clock = None
