"""Resolves where to find/save the trained weights file.

Not part of the public API (leading underscore) — an internal helper shared
by play_gui.py and app.py so both inference front ends agree on where the
model lives once this package is `pip install`-ed rather than run from a
repo checkout.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

MODEL_FILENAME = "dqn_battle_agent.pth"


def default_model_path() -> Path:
    """Return the weights file to load.

    A `dqn_battle_agent.pth` in the current working directory takes
    precedence — e.g. one you produced yourself by running the `train`
    entry point — falling back to the copy bundled inside the installed
    package (produced by the last `python train.py`/`alien-battle-train` run
    before this version was published).
    """
    local = Path.cwd() / MODEL_FILENAME
    if local.exists():
        return local
    return resources.files(__package__).joinpath(MODEL_FILENAME)
