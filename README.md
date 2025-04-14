# README for Catch Me If You Can Reinforcement Learning Project

## Overview
This project implements a simple grid-based "Catch Me If You Can" game using reinforcement learning. The agent (green) tries to avoid being caught by the enemy (red) while the enemy pursues the agent. The project uses the Proximal Policy Optimization (PPO) algorithm from Stable Baselines3 to train the agent.

## Prerequisites
- Conda (for environment management)
- Python 3.10

## Installation

1. Create and activate the Conda environment:
```bash
conda create -n catch_me python=3.10
conda activate catch_me
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Files Description
- `catch_me_env.py`: Contains the custom Gym environment for the game
- `train.py`: Script to train and test the PPO model
- `requirements.txt`: List of Python dependencies

## How to Run

1. Train the model:
```bash
python train.py
```

2. After training completes, the script will automatically switch to testing mode where you can see the trained agent in action.

3. To exit, close the Pygame window or press Ctrl+C in the terminal.

## Game Mechanics
- **Grid Size**: 10x10
- **Agent (Green)**: Controlled by the trained RL model
- **Enemy (Red)**: Moves towards the agent 80% of the time, otherwise moves randomly
- **Rewards**:
  - +0.1 for each step survived
  - -1 if caught by the enemy
- **Episode Ends**:
  - When agent is caught by enemy
  - After 100 steps (maximum episode length)

## Training Details
- Algorithm: PPO (Proximal Policy Optimization)
- Policy: MLP (Multi-Layer Perceptron)
- Total Training Timesteps: 50,000
- Model Saved As: `ppo_catch_me_if_you_can`

## Customization
You can modify the following parameters in `catch_me_env.py`:
- `grid_size`: Change the size of the game grid
- Enemy movement probability (currently 80% chance to move towards agent)
- Reward values
- Maximum episode length

## Dependencies
- Python 3.10
- numpy
- gym
- stable-baselines3
- pygame

## Contributer
- Osama Abo-Bakr
- Abdullah Abas
- Ahmed Fawzy
- Ahmed Nos7y
- Ziad Elwakel