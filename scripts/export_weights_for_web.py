"""One-time conversion: dqn_battle_agent.pth (a PyTorch state_dict) -> plain
JSON of nested Python lists, for the pygbag/WASM build.

Deliberately NOT numpy .npz: pygbag's dynamic package installer hangs
indefinitely trying to install/compile the numpy wasm wheel when it's
auto-detected as an import (reproduced during development — see git history/
conversation for the diagnostic). The web build's DQN forward pass is pure
Python (pure_dqn.py) precisely to avoid ever importing numpy in the browser,
so weights are stored as plain JSON instead.

Run this with the normal dev venv (which has torch installed) any time the
model is retrained; the web build itself never needs torch or numpy.

Usage:
    python scripts/export_weights_for_web.py
"""
from __future__ import annotations

import json

import torch

SRC = "src/dqn_alien_battle/dqn_battle_agent.pth"
DST = "web/dqn_alien_battle_web/dqn_battle_agent.json"


def main() -> None:
    state_dict = torch.load(SRC, map_location="cpu")
    weights = {k: v.tolist() for k, v in state_dict.items()}
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(weights, f)
    print(f"Wrote {DST} with keys: {list(weights.keys())}")


if __name__ == "__main__":
    main()
