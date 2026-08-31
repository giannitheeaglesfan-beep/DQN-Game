# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small reference project for learning Deep Q-Networks (DQN): a neural net learns to play a 1v1
turn-based "creature battle" game (Pokémon-lite type matchups + cooldowns) by playing thousands of
practice battles against a scripted heuristic opponent. The trained model can then be played
against via a Pygame GUI or served over a FastAPI endpoint.

## Commands

```bash
# Setup (Python 3.10+ required — code uses `list[float]` style type hints)
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Train (produces/overwrites dqn_battle_agent.pth; ~2000 episodes, a few minutes on CPU)
python train.py

# Play against the trained agent (Pygame window)
python play_gui.py

# Serve the trained model over HTTP
uvicorn app:app --reload
# POST a state vector to http://127.0.0.1:8000/predict-turn
# interactive docs at http://127.0.0.1:8000/docs
```

There is no test suite, linter, or build step configured in this repo.

`dqn_intro.ipynb` is a separate, standalone beginner tutorial (DQN on CartPole) — not part of the
creature-battle code path, and needs `matplotlib` in addition to `requirements.txt`.

## Architecture

**Shared game logic lives in `battle_env.py`, not just in the Gym env.** The module exposes
`resolve_turn()` as a standalone single-turn-resolution function specifically so that both
`CreatureBattleEnv.step()` (training) and `play_gui.py` (human play) drive the *identical* damage /
type-effectiveness / cooldown math — there is no separate "game engine" duplicated between
training and play. When changing combat rules, `resolve_turn` is the one place to change; anything
built on top (training reward shaping, the GUI's battle log) should keep working unchanged.

**Data flow across files:**
- `battle_env.py` defines the game (`CreatureBattleEnv`, `Creature`, `Move`, `resolve_turn`,
  `choose_heuristic_action`) and the 8-float observation vector (`get_observation`) — always from
  the *acting* creature's perspective (self stats first, enemy second), which is what lets one
  policy network play either side.
- `model.py` defines the learning machinery (`DQN` network, `ReplayBuffer`, `DQNAgent`) —
  independent of the game; it only knows `STATE_DIM`/`ACTION_DIM`.
- `train.py` wires the two together: runs episodes of `CreatureBattleEnv` through a `DQNAgent`
  with epsilon-greedy decay, periodically syncs the target network, and saves weights to
  `dqn_battle_agent.pth`.
- `play_gui.py` and `app.py` are two independent inference-only front ends that both load
  `dqn_battle_agent.pth` into a bare `DQN` (not a full `DQNAgent`) and pick `argmax(Q)` — no
  training code is involved at play/serve time.

**Move slots are positional, not per-creature.** Every creature (regardless of its type) has
exactly 4 moves in fixed slot semantics: 0 = Basic Attack (Normal, no cooldown), 1 = Primary STAB
(own type), 2 = Coverage (secondary type, i.e. `(type + 1) % NUM_TYPES`), 3 = Ultimate (own type,
high power/cooldown). This is what makes the action space `Discrete(4)` meaningful across
different creature type matchups — action index 1 always means "your own-type medium move" no
matter which of the 4 types is on the field. See `build_moveset()`.

**Type effectiveness is a 4-way counter cycle plus one wildcard type.** Creatures are aliens with
one of five cosmic powers (`TYPE_NAMES` in `battle_env.py`): Moon, Sun, Earth, and Meteor form a
cycle — Meteor > Earth > Sun > Moon > Meteor (`_SUPER_EFFECTIVE` / `_NOT_VERY_EFFECTIVE`). Black
Hole stands outside the cycle: neutral to and from every type, so a Black Hole alien never gets
(or takes) a super-effective hit — its fights are always "even," decided by base power/accuracy
alone, not a matchup. `ALIEN_NAMES` supplies the individual name (e.g. "Vaelthor") each creature is
given at random on top of its power — purely flavor, no mechanical effect.

**Reward shaping (`resolve_turn` + `CreatureBattleEnv.step`):** damage dealt/taken scaled to
`damage/100`, `-0.5` for a wasted turn (move on cooldown or missed), and a terminal `+10`/`-10` for
winning/losing. The opponent's own wasted-turn penalty is never applied to the agent's reward —
only actual damage the opponent lands affects the agent (see the comment in `step()`).

**`dqn_battle_agent.pth` is a committed artifact**, not just a build output — `play_gui.py` and
`app.py` both fail without it (see `app.py`'s explicit `FileNotFoundError` message). Retraining
overwrites it; if you retrain, be aware you're replacing the checkpoint the other two entry points
depend on.
