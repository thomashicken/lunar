# Lunar Lander Q-Learning Agent

This project implements a Q-learning-based agent to solve the OpenAI Gym `LunarLander-v2` environment using a neural network (Keras) as the function approximator. It includes training, evaluation, and reward tracking.

## Project Contents

- **`td-q-learning-ai-buffer.py`**  
  The main Python script implementing:
  - Q-learning with experience replay.
  - A neural network (`QFunction`) to approximate Q-values.
  - Custom reward shaping to improve landing performance.
  - Command-line options for learning (`learn`) or scoring (`score`).

- **`lunar-model.keras`**  
  A saved Keras model file containing the trained Q-network.

- **`lunar-rewards.csv`**  
  CSV log of per-epoch rewards collected during training.

## How It Works

- The agent interacts with the `LunarLander-v2` environment using:
  - An ε-greedy policy for exploration.
  - Experience replay to stabilize learning.
  - A neural network to generalize Q-values over continuous state space.

- Reward components include:
  - Proximity to the landing pad.
  - Vertical and horizontal speed penalties.
  - Penalties for tilt, excessive side engine, or main engine use.
  - Bonuses for landing legs making contact.

## How to Run

### Train the Agent:
```bash
python td-q-learning-ai-buffer.py learn --n-epochs 500 --track-epochs 1
```

### Evaluate the Trained Agent:
```bash
python td-q-learning-ai-buffer.py score --n-epochs 100
```

### Additional Options:
- `--gamma`: Discount factor (default: 0.99).
- `--batch-size`: Number of experiences per training batch.
- `--epsilon-chance-factor`: Exploration decay factor.
- `--early-stop`: Stop early if max score achieved consistently.

## Outputs

- **Model File**:  
  Trained weights saved to `lunar-model.keras` (or custom name via `--model-file`).

- **Rewards File**:  
  Per-epoch reward log saved to `lunar-rewards.csv` (or custom name via `--rewards-file`).

## Dependencies

- Python 3.x
- `gymnasium`
- `numpy`
- `tensorflow` / `keras`
- `pandas`
- `joblib`

## Notes

- The reward shaping inside `get_reward()` is critical — it boosts learning speed compared to vanilla rewards.
- Experience replay (`replay_buffer`) prevents overfitting to recent states.
- This setup focuses on discrete action space; it can be extended to continuous actions with policy gradients.
