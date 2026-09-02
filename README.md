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
serve the trained model over a small web API, or play it **in a browser** —
the same Pygame game compiled to WASM via [pygbag](https://pygame-web.github.io/),
deployable as a static site (see "Web build" below).

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

## Install from PyPI

```bash
pip install dqn-alien-battle
```

This gives you three commands, plus the package itself as an importable
library — no repo checkout needed:

```bash
alien-battle-play     # opens a Pygame window, play against the trained AI
alien-battle-serve    # runs the FastAPI server on http://127.0.0.1:8000
alien-battle-train    # retrains from scratch, writes dqn_battle_agent.pth to the current directory
```

```python
import dqn_alien_battle as dab

env = dab.CreatureBattleEnv()
state, _ = env.reset()
agent = dab.DQNAgent()
```

`alien-battle-play`/`alien-battle-serve` load the trained weights bundled
with the package by default. If you've run `alien-battle-train` yourself, a
`dqn_battle_agent.pth` in your current directory takes precedence over the
bundled one (see `src/dqn_alien_battle/_paths.py`).

## Developing from source

**Requirements:** Python 3.10+ (the code uses modern type-hint syntax like
`list[float]`).

1. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Install in editable mode:**

   ```bash
   pip install -e .
   ```

3. **Train the agent** (produces `dqn_battle_agent.pth` in the current
   directory):

   ```bash
   python -m dqn_alien_battle.train
   ```

   This runs 2000 practice battles and prints progress (average reward and
   win rate) every 100 episodes. Training takes a few minutes on a normal
   CPU — no GPU required. A pre-trained `dqn_battle_agent.pth` is already
   bundled in `src/dqn_alien_battle/`, so you can skip this step if you just
   want to play or run the API right away.

4. **Play against the trained agent** (opens a Pygame window):

   ```bash
   python -m dqn_alien_battle.play_gui
   ```

5. **Or serve the model as a web API:**

   ```bash
   uvicorn dqn_alien_battle.app:app --reload
   ```

   Then send a state vector to `http://127.0.0.1:8000/predict-turn` and it
   responds with the AI's chosen move and its Q-values for every move. Visit
   `http://127.0.0.1:8000/docs` for an interactive API explorer.

## Web build (play in a browser)

The game also runs entirely in the browser, compiled to WASM via
[pygbag](https://pygame-web.github.io/) — no server needed once deployed,
just static files. The code lives separately under `web/dqn_alien_battle_web/`
rather than reusing `src/dqn_alien_battle/` directly, because the browser
runtime can't use `torch`, `gymnasium`, or even `numpy` (see the docstrings in
`web/dqn_alien_battle_web/pure_dqn.py` and `battle_logic.py` for why — the
short version: no WASM build of torch exists, and pygbag hangs indefinitely
trying to dynamically install numpy at runtime). Everything else — game
rules, type effectiveness, move resolution — is copied verbatim from
`battle_env.py`, and the trained network's weights are exported to plain
JSON and re-implemented as a hand-rolled pure-Python forward pass, verified
numerically equivalent to the real torch model.

Build it:

```bash
pip install pygbag
python scripts/build_web.py
```

This writes a ready-to-deploy static site to
`web/dqn_alien_battle_web/build/web/`. Test it locally first:

```bash
cd web/dqn_alien_battle_web/build/web
python -m http.server 8000
# open http://localhost:8000/index.html — click the page once (browsers
# require a user gesture before audio/WASM can start)
```

Deploy that same `build/web/` folder to Vercel (or any static host) as-is:

```bash
vercel deploy web/dqn_alien_battle_web/build/web --prod
```

If you retrain the model, re-run `python scripts/export_weights_for_web.py`
(regenerates the JSON weights) and `python scripts/build_web.py` before
redeploying — the web build does not automatically pick up a new
`dqn_battle_agent.pth`.

## Files

- **`src/dqn_alien_battle/battle_env.py`** — The game itself, built as a
  [Gymnasium](https://gymnasium.farama.org/) environment (the standard
  interface RL code uses to describe "reset the game" / "take a step").
  Defines the creature types, moves, damage/type-effectiveness math,
  cooldowns, and the reward signal (damage dealt/taken, plus a bonus for
  winning and a penalty for losing). Also contains a simple rule-based
  "heuristic" opponent that the DQN trains against.

- **`src/dqn_alien_battle/model.py`** — The learning machinery: the `DQN`
  neural network (a small multi-layer perceptron), the `ReplayBuffer` that
  stores past experience for training, and the `DQNAgent` class that ties
  them together (choosing actions, learning from a batch of experience,
  saving/loading weights).

- **`src/dqn_alien_battle/train.py`** — The training loop. Runs many
  practice battles between the DQN agent and the heuristic opponent,
  gradually shifting from random moves to learned moves (epsilon-greedy
  decay), and saves the trained weights to `dqn_battle_agent.pth` (in the
  current directory) at the end. Console script: `alien-battle-train`.

- **`src/dqn_alien_battle/play_gui.py`** — A Pygame desktop app so a human
  can play against the trained model directly, with HP bars, move buttons,
  cooldown indicators, and a battle log. Console script: `alien-battle-play`.

- **`src/dqn_alien_battle/app.py`** — A small [FastAPI](https://fastapi.tiangolo.com/)
  web server that loads the trained model and exposes a `/predict-turn`
  endpoint: send it a battle state, get back the AI's chosen move. This is
  the same model used in `play_gui.py`, just accessible over HTTP instead
  of through the desktop app. Console script: `alien-battle-serve`.

- **`src/dqn_alien_battle/_paths.py`** — Resolves which `dqn_battle_agent.pth`
  to load: a local one in the current directory (e.g. one you just trained)
  takes precedence over the copy bundled inside the installed package.

- **`src/dqn_alien_battle/dqn_battle_agent.pth`** — The trained network's
  saved weights, produced by `train.py` and loaded by `play_gui.py` and
  `app.py`. Bundled as package data so `pip install dqn-alien-battle` works
  with zero setup.

- **`web/dqn_alien_battle_web/`** — The browser/pygbag build (see "Web build"
  above). `main.py` is a WASM-compatible port of `play_gui.py` (async game
  loop); `battle_logic.py` is a `torch`/`gymnasium`/`numpy`-free copy of
  `battle_env.py`'s game rules; `pure_dqn.py` is a hand-rolled pure-Python
  forward pass over the trained network, reading weights from
  `dqn_battle_agent.json`. `build/` (gitignored) holds the compiled output.

- **`scripts/export_weights_for_web.py`** — Converts `dqn_battle_agent.pth`
  (a torch state_dict) to plain JSON for the web build to load without torch.

- **`scripts/verify_pure_dqn.py`** — Confirms `pure_dqn.py`'s output matches
  the real torch model on thousands of random states before trusting it.

- **`scripts/build_web.py`** — Runs `pygbag --build` and vendors the
  `pygame-ce` wasm wheel into the output so it's self-contained for static
  hosting (pygbag's own dev server does this automatically; a plain static
  host does not).

- **`dqn_intro.ipynb`** — A standalone, beginner-friendly Jupyter notebook
  that teaches DQN from scratch using CartPole (a much simpler, classic RL
  benchmark) before you dive into the alien battle code above. Not part of
  the installable package. Needs `matplotlib` in addition to
  `requirements.txt` (`pip install matplotlib`) to run.

- **`pyproject.toml`** — Package metadata, dependencies, and the
  `alien-battle-*` console script entry points.

- **`requirements.txt`** — Convenience list of runtime dependencies for
  quick local setup; `pyproject.toml` is the canonical dependency list for
  the published package.

## Publishing a new version to PyPI

1. Bump `version` in `pyproject.toml`.
2. Build fresh distributions:
   ```bash
   rm -rf dist build src/*.egg-info
   python -m build
   twine check dist/*
   ```
3. Upload (needs a PyPI account and an API token — run `twine upload` from
   your own machine/credentials, not something to automate blindly):
   ```bash
   twine upload dist/*
   ```
   PyPI publishes are permanent per version — you cannot re-upload the same
   version number even to fix a mistake, only yank it. Consider a TestPyPI
   dry run first (`twine upload --repository testpypi dist/*`).

## Notes

- The `notebooks/` folder is excluded from this repo (see `.gitignore`) —
  it holds unrelated scratch notebooks from other practice topics, not part
  of this project.
