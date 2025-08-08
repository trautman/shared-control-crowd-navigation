# Shared Control Crowd Navigation Simulator

A Python-based simulator for evaluating shared control policies in crowded environments. Built on top of the original [crowd-navigation-simulator](https://github.com/trautman/crowd-navigation-simulator), this fork extends the architecture to support **multi-agent shared control**, where multiple decision-makers (e.g., human and AI) collaborate to navigate a robot through dense pedestrian scenarios modeled via ORCA.

---

## 📁 Repository Structure

This project lives in the `shared-control-crowd-navigation/` folder and builds upon the ORCA-based simulation infrastructure to support:

- Two simultaneous decision-makers (e.g. BRNE and DWA)
- Shared control signal blending (linear or optimal transport-based)
- Evaluation of social navigation metrics (e.g. safety distance, velocity, efficiency)
- Configurable environments, agents, goals, and robot architectures

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/shared-control-crowd-navigation.git
cd shared-control-crowd-navigation
```

### 2. Run a Simulation

```bash
python simulator.py \
    --env-config env-configs/boardwalk.yaml \
    --sim-config trials/<site>/<scenario>/RUN/simulator_config.yaml \
    --write_data_to trials/<site>/<scenario>/RUN/data/ \
    --gui
```

Replace `<site>` and `<scenario>` with your setup. The GUI shows robot trajectories, FOV, and pedestrian behavior.

---

## 🛠️ Features

- **Shared Control**: Simulates cooperation between two planners (e.g., AI and human-like agents).
- **Optimal Transport-Based Blending**: Experimental shared control using optimal transport to align joint distributions.
- **Customizable FOV**: Robot perception is restricted to configurable field of view.
- **Metrics Tracking**: Logs safety distances, density, travel time, and more.

---

## 📂 Folder Layout

```
shared-control-crowd-navigation/
├── brne_controller.py        # BRNE planner (robot)
├── dwa_controller.py         # DWA planner (human or second agent)
├── shared_control.py         # Combines decisions using blending rules
├── simulator.py              # Main simulation script
├── visualization.py          # GUI and plotting
├── env-configs/              # Static scene definitions
├── trials/                   # Simulation configs and data
└── README.md
```

---

## 📊 Output Metrics

Simulation logs include:

- Minimum distance to pedestrians
- Velocity profiles
- Travel time and path efficiency
- Social density in FOV

---

## 🤝 Contributing

This simulator is under active development. Issues, bugs, and PRs welcome!

---

## 📜 License

MIT License. See `LICENSE` file for details.