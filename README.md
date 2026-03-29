# Autonomous Maze Solver: Q-Learning from Scratch

This project builds a Reinforcement Learning maze solver from scratch using Q-Learning. The agent learns an efficient route from start to goal in a 5x5 grid while avoiding static walls.

## Overview

The project includes:
- A custom GridWorld environment implemented with NumPy
- A Q-Learning agent with epsilon-greedy exploration
- A full training + greedy policy test flow in one script
- Notebook support for plotting learning curves

## Core RL Concepts

### Bellman Update Rule

The Q-table is updated using:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[R + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]
$$

Current hyperparameters:
- Learning rate (alpha): 0.1
- Discount factor (gamma): 0.9

### Epsilon-Greedy Policy

- Initial epsilon: 1.0
- Decay per episode: epsilon = max(0.01, epsilon * 0.995)
- Minimum epsilon: 0.01

This helps the agent explore early and exploit learned behavior later.

## Environment Design

- Grid size: 5x5 (25 states)
- Actions: 4 discrete moves
  - 0 = Up
  - 1 = Down
  - 2 = Left
  - 3 = Right
- Start: (0, 0)
- Goal: (4, 4)
- Walls: (1,1), (1,2), (3,0), (3,1), (3,3)

Reward shaping:
- -1 for each step
- -5 when trying to move into a wall
- +20 when reaching the goal

## Project Structure

```text
q-learning-maze-solver/
├── train.py
├── visualize.py
├── robot.png
├── scripts/
│   ├── agent.py
│   ├── environment.py
│   └── __init__.py
├── notebook/
│   └── learning_curves.ipynb
├── data/
├── README.md
├── LICENSE
└── .gitignore
```

## Run the Project

### 1. Install dependencies

```bash
pip install numpy matplotlib jupyter pygame
```

### 2. Train and test policy

```bash
python train.py
```

The script does two phases:
1. Train for 1000 episodes and print reward every 100 episodes
2. Run a greedy test (pure exploitation) and print step-by-step moves in text form

Example move output:
- Step 1: Agent move Right
- Step 2: Agent move Right

## Understanding Output Terms

- Step: One move attempt in the maze
- Action: Numeric move selected by the policy
- Agent move: Human-readable direction label (Up, Down, Left, Right)

If the agent reaches the goal in 8 steps without wall collisions, a final reward near 13 is expected in the last episode of stable learning.

## Notebook Usage

- Use notebook/learning_curves.ipynb to visualize rewards and inspect the Q-table

For reproducible training, use train.py as the main entry point.

## Visualization Script

- `visualize.py` is available for Pygame-based rendering of the agent in the maze.
- `robot.png` is used as the agent sprite.

## Author

Kavindya Ranaweera
Focus: AI/ML Engineering and Robotics
Portfolio: Part of a 3-project series (Data Engineering, Financial Quants, and RL)

## License

This project is licensed under the MIT License.