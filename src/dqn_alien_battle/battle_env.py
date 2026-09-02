"""Custom Gymnasium environment for a 1v1 turn-based alien battle game.

Aliens wield one of five cosmic powers. Moon, Sun, Earth, and Meteor form a
4-way counter cycle (Meteor > Earth > Sun > Moon > Meteor). Black Hole sits
outside that cycle entirely: it's never super effective and never weak to
anything, so a Black Hole alien's fights are always "even" — no free
matchup advantage either way, just raw power and accuracy.

RL framing:
    - Observation: 8-float vector normalized to [0, 1], always from the
      *acting* creature's own perspective (self stats first, enemy second).
    - Action: Discrete(4), selecting one of the acting creature's 4 fixed
      move slots.
    - Reward: shaped by damage dealt/taken plus a terminal +/-10 for
      winning/losing, with a small penalty for wasted turns (move on
      cooldown or a miss).

This module intentionally exposes a single-turn-resolution helper
(`resolve_turn`) instead of only a monolithic `step()`, so that other code
(e.g. the Pygame GUI) can drive one side of the battle with a trained model
instead of the built-in heuristic opponent, while guaranteeing identical
damage / type-effectiveness / cooldown math in both training and play.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

NUM_TYPES = 5
NUM_ELEMENTAL_TYPES = 4  # Moon/Sun/Earth/Meteor take part in the counter cycle; Black Hole stands apart
TYPE_MOON, TYPE_SUN, TYPE_EARTH, TYPE_METEOR, TYPE_BLACKHOLE = range(NUM_TYPES)
TYPE_NAMES = ["Moon", "Sun", "Earth", "Meteor", "Black Hole"]

# Pool of alien names handed out to creatures at creation, independent of
# their power. Extend this list anytime — battles pick uniformly at random.
ALIEN_NAMES = [
    "Astra", "Orion", "Nova", "Cosmo", "Vega",
    "Xan", "Vaelthor", "Kalthuun", "Orivex", "Zeraphon",
    "Tala", "Izar", "Euna", "Elazar", "Alistir",
]

MAX_HP = 100
NUM_MOVES = 4
MAX_COOLDOWN = 3  # used only for observation normalization (cd / 3)
MAX_TURNS = 50  # safety cap so episodes always terminate

# Move `power` values (40/70/55/100) are base stats, not raw damage — at 1:1
# scale a single 100-power Ultimate could nearly one-shot a 100-HP creature,
# which ends fights in 1-2 turns instead of a multi-turn back-and-forth.
# This scales base power down to an actual HP delta so a typical battle runs
# for several exchanges per side.
DAMAGE_SCALE = 0.3

# Type effectiveness: a 4-way counter cycle among the elemental powers —
# Meteor > Earth > Sun > Moon > Meteor (rock-paper-scissors-lizard).
# Black Hole is neutral against everything and everything is neutral against
# it — it never gets a super-effective hit and is never caught by one, so a
# Black Hole alien's matchups are always even; only its base power/accuracy
# decides the fight.
_SUPER_EFFECTIVE = {
    (TYPE_METEOR, TYPE_EARTH): 2.0,
    (TYPE_EARTH, TYPE_SUN): 2.0,
    (TYPE_SUN, TYPE_MOON): 2.0,
    (TYPE_MOON, TYPE_METEOR): 2.0,
}
_NOT_VERY_EFFECTIVE = {
    (TYPE_EARTH, TYPE_METEOR): 0.5,
    (TYPE_SUN, TYPE_EARTH): 0.5,
    (TYPE_MOON, TYPE_SUN): 0.5,
    (TYPE_METEOR, TYPE_MOON): 0.5,
}


def type_effectiveness(attack_type: int, defend_type: int) -> float:
    """Return the damage multiplier for attack_type hitting defend_type."""
    if attack_type == TYPE_BLACKHOLE or defend_type == TYPE_BLACKHOLE:
        return 1.0
    key = (attack_type, defend_type)
    if key in _SUPER_EFFECTIVE:
        return 2.0
    if key in _NOT_VERY_EFFECTIVE:
        return 0.5
    return 1.0


@dataclass
class Move:
    name: str
    power: int
    accuracy: float
    cooldown: int
    move_type: int


def build_moveset(creature_type: int) -> list[Move]:
    """Fixed 4-slot move schema shared by every creature of a given power.

    Slot semantics are identical across all creatures so action index 0-3
    always means the same *kind* of move to the DQN, regardless of which
    power is on the field:
        0: Basic Attack   - Black Hole type (always neutral), no cooldown.
        1: Primary STAB   - matches own power, medium power, 1 cd.
        2: Coverage       - secondary power, medium power, 1 cd.
        3: Ultimate       - matches own power, high power, 2 cd, riskier.

    Black Hole aliens have no elemental cycle to draw a secondary power
    from, so every one of their moves is Black Hole type — they never land
    (or take) a super-effective hit; their edge is raw power/accuracy, not
    a matchup.
    """
    if creature_type == TYPE_BLACKHOLE:
        secondary_type = TYPE_BLACKHOLE
        own_name = "Black Hole"
        secondary_name = "Void"  # flavor only — move_type below is still Black Hole
    else:
        secondary_type = (creature_type + 1) % NUM_ELEMENTAL_TYPES
        own_name = TYPE_NAMES[creature_type]
        secondary_name = TYPE_NAMES[secondary_type]
    return [
        Move(name="Basic Attack", power=40, accuracy=0.95, cooldown=0, move_type=TYPE_BLACKHOLE),
        Move(name=f"{own_name} Strike", power=70, accuracy=0.90, cooldown=1, move_type=creature_type),
        Move(name=f"{secondary_name} Coverage", power=55, accuracy=0.90, cooldown=1, move_type=secondary_type),
        Move(name=f"{own_name} Ultimate", power=100, accuracy=0.80, cooldown=2, move_type=creature_type),
    ]


@dataclass
class Creature:
    creature_type: int
    name: str = ""
    hp: int = MAX_HP
    moves: list[Move] = field(default_factory=list)
    cooldowns: list[int] = field(default_factory=lambda: [0] * NUM_MOVES)

    @property
    def hp_pct(self) -> float:
        return max(0.0, self.hp) / MAX_HP

    def is_fainted(self) -> bool:
        return self.hp <= 0


def make_creature(creature_type: int, rng: random.Random | None = None) -> Creature:
    """Create a creature of the given power with a random alien name.

    `rng` is optional so callers without a seeded `random.Random` (e.g. quick
    scripts) can still call this; `CreatureBattleEnv` always passes its own
    seeded instance through, so name assignment stays reproducible.
    """
    picker = rng if rng is not None else random
    name = picker.choice(ALIEN_NAMES)
    return Creature(creature_type=creature_type, name=name, hp=MAX_HP, moves=build_moveset(creature_type))


def get_observation(actor: Creature, opponent: Creature) -> np.ndarray:
    """Build the normalized 8-float state vector from `actor`'s perspective."""
    return np.array(
        [
            actor.hp_pct,
            opponent.hp_pct,
            actor.creature_type / (NUM_TYPES - 1),
            opponent.creature_type / (NUM_TYPES - 1),
            actor.cooldowns[0] / MAX_COOLDOWN,
            actor.cooldowns[1] / MAX_COOLDOWN,
            actor.cooldowns[2] / MAX_COOLDOWN,
            actor.cooldowns[3] / MAX_COOLDOWN,
        ],
        dtype=np.float32,
    )


def resolve_turn(attacker: Creature, defender: Creature, action: int, rng: random.Random) -> tuple[float, str]:
    """Resolve one creature's chosen move against another.

    Returns (reward_delta, log_message). Applies damage/cooldown mutations
    in place on `attacker` and `defender`. Does not check for faint/win —
    callers are responsible for checking HP after calling this.
    """
    move = attacker.moves[action]

    if attacker.cooldowns[action] > 0:
        return -0.5, f"{attacker.name}'s {move.name} is on cooldown!"

    if rng.random() > move.accuracy:
        attacker.cooldowns[action] = move.cooldown
        return -0.5, f"{attacker.name}'s {move.name} missed!"

    multiplier = type_effectiveness(move.move_type, defender.creature_type)
    damage = move.power * multiplier * DAMAGE_SCALE
    defender.hp = max(0, defender.hp - round(damage))
    attacker.cooldowns[action] = move.cooldown

    effectiveness_note = ""
    if multiplier > 1.0:
        effectiveness_note = " It's super effective!"
    elif multiplier < 1.0:
        effectiveness_note = " It's not very effective..."

    return damage / 100.0, f"{attacker.name}'s {move.name} hit for {round(damage)} damage!{effectiveness_note}"


def choose_heuristic_action(actor: Creature, opponent: Creature, rng: random.Random) -> int:
    """Simple opponent AI: prefer the highest-power available move that has
    type advantage over the opponent; falls back to any available move."""
    available = [i for i in range(NUM_MOVES) if actor.cooldowns[i] == 0]
    if not available:
        return rng.randrange(NUM_MOVES)

    def score(i: int) -> tuple[float, int]:
        move = actor.moves[i]
        mult = type_effectiveness(move.move_type, opponent.creature_type)
        return (mult, move.power)

    return max(available, key=score)


class CreatureBattleEnv(gym.Env):
    """1v1 turn-based creature battle environment.

    Each call to `step(action)` resolves the learning agent's move, then
    (if the opponent survives) the built-in heuristic opponent's move.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_MOVES)
        self._rng = random.Random(seed)
        self.agent: Creature
        self.opponent: Creature
        self._turn_count = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        agent_type = self._rng.randrange(NUM_TYPES)
        opponent_type = self._rng.randrange(NUM_TYPES)
        self.agent = make_creature(agent_type, self._rng)
        self.opponent = make_creature(opponent_type, self._rng)
        self._turn_count = 0
        return get_observation(self.agent, self.opponent), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward, log = resolve_turn(self.agent, self.opponent, int(action), self._rng)
        info = {"log": [log]}

        if self.opponent.is_fainted():
            reward += 10.0
            info["log"].append("You win!")
            self._decrement_cooldowns()
            return get_observation(self.agent, self.opponent), reward, True, False, info

        opp_action = choose_heuristic_action(self.opponent, self.agent, self._rng)
        opp_reward, opp_log = resolve_turn(self.opponent, self.agent, opp_action, self._rng)
        # opp_reward is the opponent's own reward signal (damage/100, or -0.5 for
        # its own miss/cooldown). Only damage actually dealt to the agent should
        # affect the agent's reward — the opponent fumbling its turn is not the
        # agent's concern.
        if opp_reward > 0:
            reward -= opp_reward
        info["log"].append(opp_log)

        terminated = False
        if self.agent.is_fainted():
            reward -= 10.0
            info["log"].append("You lose!")
            terminated = True

        self._decrement_cooldowns()
        self._turn_count += 1
        truncated = self._turn_count >= MAX_TURNS and not terminated

        return get_observation(self.agent, self.opponent), reward, terminated, truncated, info

    def _decrement_cooldowns(self) -> None:
        for creature in (self.agent, self.opponent):
            creature.cooldowns = [max(0, cd - 1) for cd in creature.cooldowns]

    def render(self) -> None:  # pragma: no cover - visualization handled by play_gui.py
        pass
