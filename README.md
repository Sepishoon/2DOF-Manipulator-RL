# 2DOF Manipulator RL

A comparative study of **classical control** and **deep reinforcement learning** for a 2-DOF planar robotic arm, implemented as a custom Gymnasium environment.

> **Course:** Reinforcement Learning — Fall 2025  
> **Department:** Aerospace Engineering, Sharif University of Technology  
> **Author:** Sepehr Mahfar

---

## Overview

This project implements and compares four control strategies for a two-link planar reacher operating in a horizontal plane:

| Method | Mean Tracking Error |
|---|---|
| Cartesian PD Control | 0.973 |
| IK + Joint PD Control | 0.987 |
| N-Step SARSA | 1.773 |
| DDPG (100 ep) | 1.369 |
| DDPG (1000 ep, Config C) | **0.90** |

---

## Environment

The custom `ReacherEnv` (Gymnasium-compatible) simulates a 2-DOF planar arm with the following parameters:

| Parameter | Value |
|---|---|
| Link lengths | 1.0 m each |
| Link masses | 1.0 kg each |
| Joint damping | 0.05 kg·m/s |
| Time step | 0.02 s |
| Max steps/episode | 150 (3 seconds) |
| Torque bounds | [−1, 1] N·m per joint |

**Observation space:** `(x_target, y_target, θ₁, θ₂, θ̇₁, θ̇₂)` — 6-dimensional  
**Reward:** Negative squared Euclidean distance to target

---

## Project Structure

```
.
├── reacher_env.py               # Custom Gymnasium environment (Part a)
├── test_env.py                  # Environment sanity checks
├── play_pygame.py               # Interactive manual control via Pygame
│
├── part_b_pid_cartesian.py      # Cartesian-space PD control
├── part_c_pid_ik.py             # Inverse kinematics + joint-space PD control
├── part_d_sarsa.py              # N-Step SARSA with discrete actions
│
├── part_e_ddpg_100ep.py         # DDPG — Config A (100 episodes)
├── part_e_ddpg.py               # DDPG — Config B (1000 episodes)
├── part_e_ddpg_custom_configuration.py  # DDPG — Config C (custom protocol)
│
├── part_f_noise_study.py        # Exploration noise comparison (OU vs Gaussian vs None)
├── part_g_generalization.py     # Generalization tests under distribution shift
├── part_h_ablation.py           # Ablation study on state representations
├── part_i_metrics.py            # Final metrics comparison across all methods
│
├── Videos/                      # Recorded rollout videos
└── Report.pdf                   # Full project report
```

---

## Methods

### Classical Control
- **Part b — Cartesian PD:** Task-space PID via Jacobian transpose mapping. Final gains: `Kp=10, Kd=8` (integral term removed as harmful for 3-second episodes).
- **Part c — IK + Joint PD:** Analytical inverse kinematics followed by independent joint-space PD controllers. Final gains: `Kp=20, Kd=10`.

### Reinforcement Learning
- **Part d — N-Step SARSA:** On-policy RL over a discrete 9-action space `{-1, 0, +1}²`. Limited by coarse discretization causing bang-bang behavior (smoothness: 43.3).
- **Part e — DDPG:** Off-policy actor-critic with continuous torque output `[-1,1]²`. Three configurations investigated:
  - **Config A** (100 ep, lightweight 64→32): Error 1.43
  - **Config B** (1000 ep, 256→128): Error 1.15
  - **Config C** (custom protocol, deferred learning, burst updates, ELU): Error **0.90** — surpasses both PID controllers

### Additional Studies
- **Part f — Noise Study:** OU noise significantly outperforms Gaussian noise and no noise (10.5% vs ~6% success rate). The structure of exploration noise matters more than its mere presence.
- **Part g — Generalization:** Agent generalizes moderately to speed changes but degrades sharply under spatial distribution shift and fails on static targets (never seen during training).
- **Part h — Ablation:** With sufficient training, DDPG is robust to state representation choice. Error-only (6-dim) performs best; IK-compressed state (4-dim) is 16% worse.
- **Part i — Metrics:** Standardized comparison across tracking error, success rate, control energy, and smoothness.

---

## Installation

```bash
pip install gymnasium numpy torch pygame matplotlib
```

---

## Usage

```bash
# Test the environment
python test_env.py

# Play manually with keyboard
python play_pygame.py

# Run classical controllers
python part_b_pid_cartesian.py
python part_c_pid_ik.py

# Train and evaluate RL agents
python part_d_sarsa.py
python part_e_ddpg.py

# Run studies
python part_f_noise_study.py
python part_g_generalization.py
python part_h_ablation.py
python part_i_metrics.py
```

---

## Key Findings

1. **Model knowledge enables precision** — classical PD controllers outperform limited-budget RL, but DDPG closes ~55% of the gap with 1000 episodes of training.
2. **Continuous actions are essential** — DDPG is 18× smoother than SARSA and achieves 23% lower tracking error.
3. **OU noise structure matters** — temporally correlated exploration roughly doubles success rate vs. Gaussian noise.
4. **100 episodes is a severe constraint** — RL methods require far more data than is typical in standard benchmarks to reach competitive performance.
5. **Generalization is bounded by training diversity** — training on diverse target distributions would likely improve out-of-distribution performance significantly.
