# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small reference project for learning Deep Q-Networks (DQN): a neural net learns to play a 1v1
turn-based "creature battle" game (Pokémon-lite type matchups + cooldowns) by playing thousands of
practice battles against a scripted heuristic opponent. The trained model can then be played
against via a Pygame GUI or served over a FastAPI endpoint.

## Commands

This is now a proper installable package (`dqn-alien-battle` on PyPI), source under `src/dqn_alien_battle/`.

```bash
# Setup (Python 3.10+ required — code uses `list[float]` style type hints)
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .              # editable install from pyproject.toml

# Train (writes/overwrites ./dqn_battle_agent.pth; ~2000 episodes, a few minutes on CPU)
python -m dqn_alien_battle.train          # or: alien-battle-train

# Play against the trained agent (Pygame window)
python -m dqn_alien_battle.play_gui       # or: alien-battle-play

# Serve the trained model over HTTP
uvicorn dqn_alien_battle.app:app --reload # or: alien-battle-serve
# POST a state vector to http://127.0.0.1:8000/predict-turn
# interactive docs at http://127.0.0.1:8000/docs

# Build/validate a release (does NOT upload — that's a manual `twine upload dist/*`)
rm -rf dist build src/*.egg-info && python -m build && twine check dist/*

# Build the browser/WASM version (pygbag) — output ready to deploy as a static site
pip install pygbag
python scripts/build_web.py    # writes web/dqn_alien_battle_web/build/web/
```

There is no test suite or linter configured in this repo.

`dqn_intro.ipynb` (repo root) is a separate, standalone beginner tutorial (DQN on CartPole) — not
part of the `dqn_alien_battle` package, and needs `matplotlib` in addition to its dependencies.

## Architecture

**Package layout:** everything importable lives under `src/dqn_alien_battle/` (standard src-layout,
declared via `[tool.setuptools.packages.find] where = ["src"]` in `pyproject.toml`). `__init__.py`
re-exports the game/learning public API (`CreatureBattleEnv`, `DQNAgent`, etc.); `train.py`,
`play_gui.py`, and `app.py` are entry-point scripts, not re-exported, since each pulls in
heavier dependencies (pygame, fastapi/uvicorn) — import them explicitly
(`from dqn_alien_battle import app`) or run them via their `alien-battle-*` console scripts
(`[project.scripts]` in `pyproject.toml`). `_paths.py` (leading underscore = internal, not public
API) resolves which `dqn_battle_agent.pth` to load: a local one in the current working directory
(e.g. freshly retrained) takes precedence over the copy bundled inside the installed package via
`[tool.setuptools.package-data]`. `train.py` always *writes* to the current directory regardless —
retraining never overwrites the package's own bundled copy in `site-packages`.

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
  `dqn_battle_agent.pth` (in the current directory — see package layout note above).
- `play_gui.py` and `app.py` are two independent inference-only front ends that both load
  `dqn_battle_agent.pth` (via `_paths.default_model_path()`) into a bare `DQN` (not a full
  `DQNAgent`) and pick `argmax(Q)` — no training code is involved at play/serve time.

**Move slots are positional, not per-creature.** Every creature (regardless of its type) has
exactly 4 moves in fixed slot semantics: 0 = Basic Attack (Black Hole type, always neutral, no
cooldown), 1 = Primary STAB (own type), 2 = Coverage (secondary type, i.e.
`(type + 1) % NUM_ELEMENTAL_TYPES` for the 4 elemental types — Black Hole creatures get all-Black-Hole
moves instead, see `build_moveset()`'s special case), 3 = Ultimate (own type, high power/cooldown).
This is what makes the action space `Discrete(4)` meaningful across different creature type
matchups — action index 1 always means "your own-power medium move" no matter which power is on
the field.

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

**`dqn_battle_agent.pth` is a committed artifact**, not just a build output — it's bundled as
package data (see `pyproject.toml`'s `[tool.setuptools.package-data]` and `MANIFEST.in`) precisely
so `pip install dqn-alien-battle` works with zero setup. `play_gui.py`/`app.py` both fail without
*some* copy of it resolvable (see `app.py`'s explicit `FileNotFoundError` message). If you retrain
locally, the new file lands in your current directory and is preferred over the bundled copy (see
`_paths.py`) — but the bundled copy only updates when you rebuild and republish the package.

**The web build (`web/dqn_alien_battle_web/`) is a deliberate fork of the game logic, not the same
files reused.** It exists to compile via pygbag to WASM for browser play, and three of this
project's normal dependencies are impossible or broken there, confirmed by hands-on testing during
development, not assumption:
- **`torch` has no WASM build at all** — inference is instead a hand-rolled pure-Python forward
  pass (`pure_dqn.py`) over weights exported to plain JSON (`scripts/export_weights_for_web.py`),
  checked numerically equivalent to the real torch model (`scripts/verify_pure_dqn.py`) before
  being trusted.
- **`numpy`, if imported anywhere in the web build, hangs pygbag indefinitely** — pygbag
  auto-detects the import and tries to dynamically install/compile a numpy wasm wheel at runtime;
  this was reproduced getting stuck forever at "Scanning ... for WebAssembly libraries [compiling]"
  with a completely blank canvas and no error. `battle_logic.py` and `pure_dqn.py` are numpy-free
  for this reason — don't reintroduce `import numpy` into anything under `web/` without retesting
  that path.
- **`gymnasium` (`CreatureBattleEnv`) isn't needed to play, only to train** — `battle_logic.py` is
  a copy of `battle_env.py`'s non-training parts (constants, `Move`/`Creature`, `build_moveset`,
  `type_effectiveness`, `resolve_turn`, `choose_heuristic_action`); `play_gui.py` already only
  calls those free functions directly rather than `CreatureBattleEnv.step()`, so this was a
  drop-in swap. If you change combat rules in `battle_env.py`, mirror the change in
  `battle_logic.py` too — this is the one deliberate, documented exception to "`resolve_turn` is
  the single source of truth."

**`pygbag --build` alone does not produce a deployable static site.** The generated page fetches
any package beyond the interpreter itself (here, `pygame-ce`) from a same-origin relative path
(`/cdn/cp312/<wheel>`) that only pygbag's own local dev server serves automatically — a plain
static host (Vercel included) 404s on it, leaving the game frozen on a blank grey canvas with no
visible error. `scripts/build_web.py` fetches the real wheel from the pygame-web CDN once and
vendors a copy into the build output at that exact path so the deployed site is self-contained;
confirmed by actually loading the built output in a browser, not just by the build succeeding.
