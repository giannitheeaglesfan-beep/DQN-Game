# DQN Creature Battle

A small project for learning **Deep Q-Networks (DQN)**, a reinforcement learning
algorithm. A neural network learns to play a simple 1v1 turn-based alien
battle game (pick a move each turn, exploit power matchups, watch out for
cooldowns) purely by playing thousands of practice battles against a scripted
opponent.

Each alien wields one of five cosmic powers — Moon, Sun, Earth, and Meteor
form a counter cycle (Meteor > Earth > Sun > Moon > Meteor), while Black Hole
stands outside it: never super effective, never weak to anything, so its
fights always come down to raw power and accuracy rather than a matchup.

Once trained, you can either play against the AI yourself in a Pygame window,
or serve the trained model over a small web API.

## Why this project exists

This is a reference/practice project for learning RL concepts hands-on:

- **Environment design** — how to turn a game's rules into an
  observation vector, an action space, and a reward signal.
- **Experience replay** — why RL agents learn from a buffer of past
  experience instead of only the most recent step.
- **Target networks** — why DQN uses a second, slower-updating copy of
  the network to keep training stable.
- **Epsilon-greedy exploration** — how an agent balances trying random
  moves (exploration) against using what it's already learned
  (exploitation).

If you're new to RL, start with `dqn_intro.ipynb` (see below) before
reading `model.py` and `battle_env.py`.

## Setup

**Requirements:** Python 3.10+ (the code uses modern type-hint syntax like
`list[float]`).

1. **Create and activate a virtual environment** (recommended so these
   packages don't clash with anything else on your machine):

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the agent** (produces `dqn_battle_agent.pth`, the trained
   model's weights):

   ```bash
   python train.py
   ```

   This runs 2000 practice battles and prints progress (average reward and
   win rate) every 100 episodes. Training takes a few minutes on a normal
   CPU — no GPU required. A pre-trained `dqn_battle_agent.pth` is already
   included in this repo, so you can skip this step if you just want to
   play or run the API right away.

4. **Play against the trained agent** (opens a Pygame window):

   ```bash
   python play_gui.py
   ```

5. **Or serve the model as a web API:**

   ```bash
   uvicorn app:app --reload
   ```

   Then send a state vector to `http://127.0.0.1:8000/predict-turn` and it
   responds with the AI's chosen move and its Q-values for every move. Visit
   `http://127.0.0.1:8000/docs` for an interactive API explorer.

## Files

- **`battle_env.py`** — The game itself, built as a
  [Gymnasium](https://gymnasium.farama.org/) environment (the standard
  interface RL code uses to describe "reset the game" / "take a step").
  Defines the creature types, moves, damage/type-effectiveness math,
  cooldowns, and the reward signal (damage dealt/taken, plus a bonus for
  winning and a penalty for losing). Also contains a simple rule-based
  "heuristic" opponent that the DQN trains against.

- **`model.py`** — The learning machinery: the `DQN` neural network
  (a small multi-layer perceptron), the `ReplayBuffer` that stores past
  experience for training, and the `DQNAgent` class that ties them together
  (choosing actions, learning from a batch of experience, saving/loading
  weights).

- **`train.py`** — The training loop. Runs many practice battles between
  the DQN agent and the heuristic opponent, gradually shifting from random
  moves to learned moves (epsilon-greedy decay), and saves the trained
  weights to `dqn_battle_agent.pth` at the end.

- **`play_gui.py`** — A Pygame desktop app so a human can play against the
  trained model directly, with HP bars, move buttons, cooldown indicators,
  and a battle log. Requires a trained `dqn_battle_agent.pth`.

- **`app.py`** — A small [FastAPI](https://fastapi.tiangolo.com/) web
  server that loads the trained model and exposes a `/predict-turn`
  endpoint: send it a battle state, get back the AI's chosen move. This is
  the same model used in `play_gui.py`, just accessible over HTTP instead
  of through the desktop app.

- **`dqn_intro.ipynb`** — A standalone, beginner-friendly Jupyter notebook
  that teaches DQN from scratch using CartPole (a much simpler, classic RL
  benchmark) before you dive into the creature battle code above. Needs
  `matplotlib` in addition to `requirements.txt` (`pip install matplotlib`)
  to run.

- **`dqn_battle_agent.pth`** — The trained network's saved weights,
  produced by `train.py` and loaded by `play_gui.py` and `app.py`.

- **`requirements.txt`** — Python package dependencies.

## Notes

- The `notebooks/` folder is excluded from this repo (see `.gitignore`) —
  it holds unrelated scratch notebooks from other practice topics, not part
  of this project.
