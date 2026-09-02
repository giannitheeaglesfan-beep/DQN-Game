"""dqn_alien_battle — a Deep Q-Network learns to play a 1v1 turn-based alien battle game.

Public API re-exported here is the surface meant for `import dqn_alien_battle`
usage in other people's code. The game/learning internals are documented in
their own modules (`battle_env`, `model`); `train`, `play_gui`, and `app` are
entry-point scripts rather than library API and are imported explicitly
(`from dqn_alien_battle import app` etc.) rather than re-exported here, since
each pulls in heavier optional-feeling dependencies (pygame, fastapi/uvicorn).
"""

from .battle_env import (
    ALIEN_NAMES,
    NUM_TYPES,
    TYPE_NAMES,
    Creature,
    CreatureBattleEnv,
    Move,
    choose_heuristic_action,
    get_observation,
    resolve_turn,
    type_effectiveness,
)
from .model import ACTION_DIM, STATE_DIM, DQN, DQNAgent, ReplayBuffer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # battle_env
    "ALIEN_NAMES",
    "NUM_TYPES",
    "TYPE_NAMES",
    "Creature",
    "CreatureBattleEnv",
    "Move",
    "choose_heuristic_action",
    "get_observation",
    "resolve_turn",
    "type_effectiveness",
    # model
    "ACTION_DIM",
    "STATE_DIM",
    "DQN",
    "DQNAgent",
    "ReplayBuffer",
]
