# Shared-Control Crowd Navigation Simulator

This repository extends the [crowd-navigation-simulator](https://github.com/trautman/crowd-navigation-simulator) framework to support **shared control** between a robot autonomy module and a human (or simulated human) co-controller navigating through dense pedestrian crowds.

## 🔧 Core Foundation

The core of this simulator is the original `crowd-navigation-simulator`, which:
- Simulates pedestrian motion using **Optimal Reciprocal Collision Avoidance (ORCA)**
- Uses either **BRNE** (Bayesian Recursion for Nash Equilibrium) or **DWA** (Dynamic Window Approach) for autonomous robot control

This repository **freezes** that codebase and builds additional shared-control logic on top of it.

---

## 🤝 Shared Control Architecture

In standard operation:
- **BRNE** (autonomy) observes pedestrians in the robot’s field of view (FOV), plans a safe trajectory, and outputs velocity commands.
- The simulator advances one time step and repeats.

In shared-control operation:
1. **Two decision-makers** observe pedestrians in the FOV:
   - `BRNE` (robot autonomy)
   - `DWA` (simulated human or user model)
2. Each produces a velocity command.
3. A **shared controller** (e.g. linear blending or optimal transport-based policy) combines both commands into a final control signal for the robot.

---

## 🗂️ Repository Structure

```plaintext
shared-control-crowd-navigation/
├── crowd-navigation-simulator/     # Frozen core simulator
├── shared_control/                 # Shared control logic (blending, controller classes, etc.)
├── sim_prompt_freeze_task/         # Change-tracking system with documented development tasks
├── README.md
└── ...
