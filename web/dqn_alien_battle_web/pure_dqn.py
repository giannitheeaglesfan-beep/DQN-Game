"""Pure-Python (no numpy, no torch) re-implementation of model.py's
DQN.forward(), for the pygbag web build.

Why not numpy: pygbag auto-detects `import numpy` and tries to dynamically
install/compile the numpy wasm wheel at runtime — this was reproduced to hang
indefinitely (stuck at "Scanning ... for WebAssembly libraries [compiling]"
with zero progress for 3+ minutes) during development. The network here is
tiny (8 -> 128 -> 128 -> 4), so a hand-rolled list-based forward pass is both
simpler and removes an entire fragile dependency for a trivial amount of
compute. Weights come from a plain JSON export (scripts/export_weights_for_web.py)
of the same trained torch state_dict, verified numerically equivalent in
scripts/verify_pure_dqn.py before being trusted here.
"""
from __future__ import annotations

import json


def _matvec_relu(weights: list[list[float]], bias: list[float], x: list[float]) -> list[float]:
    """y = relu(W @ x + b), everything as plain Python lists."""
    out = []
    for row, b in zip(weights, bias):
        s = b
        for w, xi in zip(row, x):
            s += w * xi
        out.append(s if s > 0.0 else 0.0)
    return out


def _matvec(weights: list[list[float]], bias: list[float], x: list[float]) -> list[float]:
    """y = W @ x + b, no activation (final layer)."""
    out = []
    for row, b in zip(weights, bias):
        s = b
        for w, xi in zip(row, x):
            s += w * xi
        out.append(s)
    return out


class PureDQN:
    def __init__(self, json_path: str) -> None:
        with open(json_path, encoding="utf-8") as f:
            weights = json.load(f)
        self.w1 = weights["net.0.weight"]  # 128 x 8
        self.b1 = weights["net.0.bias"]  # 128
        self.w2 = weights["net.2.weight"]  # 128 x 128
        self.b2 = weights["net.2.bias"]  # 128
        self.w3 = weights["net.4.weight"]  # 4 x 128
        self.b3 = weights["net.4.bias"]  # 4

    def q_values(self, state: list[float]) -> list[float]:
        """state: 8 floats -> returns 4 Q-values, one per move."""
        h1 = _matvec_relu(self.w1, self.b1, state)
        h2 = _matvec_relu(self.w2, self.b2, h1)
        return _matvec(self.w3, self.b3, h2)

    def best_action(self, state: list[float]) -> int:
        q = self.q_values(state)
        return q.index(max(q))
