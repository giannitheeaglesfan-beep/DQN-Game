"""Game logic for the pygbag web build — a straight copy of the non-training
parts of src/dqn_alien_battle/battle_env.py (constants, Move/Creature,
build_moveset, type_effectiveness, resolve_turn, choose_heuristic_action),
minus the `gymnasium`-based `CreatureBattleEnv` class.

Why a separate copy instead of importing battle_env.py directly: pygbag/
Pyodide has no reliable WASM build of `gymnasium` to import, and it isn't
needed to *play* the game anyway — play_gui.py (the desktop version this web
build is ported from) already only calls the free functions below directly
(`resolve_turn`, `choose_heuristic_action`) rather than going through
`CreatureBattleEnv.step()`. If you change combat rules in battle_env.py,
mirror the change here too — see CLAUDE.md's note on `resolve_turn` being
the single source of truth for combat math; this file is the one deliberate,
documented exception, forced by the WASM runtime rather than a design choice.

Also deliberately no numpy here (unlike battle_env.py): pygbag auto-detects
`import numpy` and tries to dynamically install/compile it at runtime, which
was reproduced to hang indefinitely during development. get_observation
below returns a plain list instead of an np.array for this reason.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

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
    always means the same *kind* of move, regardless of which power is on
    the field:
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
    """Create a creature of the given power with a random alien name."""
    picker = rng if rng is not None else random
    name = picker.choice(ALIEN_NAMES)
    return Creature(creature_type=creature_type, name=name, hp=MAX_HP, moves=build_moveset(creature_type))


def get_observation(actor: Creature, opponent: Creature) -> list[float]:
    """Build the normalized 8-float state vector from `actor`'s perspective."""
    return [
        actor.hp_pct,
        opponent.hp_pct,
        actor.creature_type / (NUM_TYPES - 1),
        opponent.creature_type / (NUM_TYPES - 1),
        actor.cooldowns[0] / MAX_COOLDOWN,
        actor.cooldowns[1] / MAX_COOLDOWN,
        actor.cooldowns[2] / MAX_COOLDOWN,
        actor.cooldowns[3] / MAX_COOLDOWN,
    ]


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
