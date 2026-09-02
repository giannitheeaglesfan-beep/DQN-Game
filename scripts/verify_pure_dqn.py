"""Confirm PureDQN (pure Python, no numpy/torch) produces the same output as
the real torch DQN on many random states, before trusting it for the web
build."""
from __future__ import annotations

import random
import sys

import torch

sys.path.insert(0, "web/dqn_alien_battle_web")
sys.path.insert(0, "src")

from pure_dqn import PureDQN  # noqa: E402
from dqn_alien_battle.model import DQN, STATE_DIM  # noqa: E402

torch_model = DQN()
torch_model.load_state_dict(torch.load("src/dqn_alien_battle/dqn_battle_agent.pth", map_location="cpu"))
torch_model.eval()

pure_model = PureDQN("web/dqn_alien_battle_web/dqn_battle_agent.json")

rng = random.Random(0)
max_abs_diff = 0.0
mismatched_actions = 0
N = 2000
for _ in range(N):
    state = [rng.random() for _ in range(STATE_DIM)]

    with torch.no_grad():
        torch_q = torch_model(torch.tensor(state).unsqueeze(0)).squeeze(0).tolist()
    pure_q = pure_model.q_values(state)

    max_abs_diff = max(max_abs_diff, max(abs(a - b) for a, b in zip(torch_q, pure_q)))
    if torch_q.index(max(torch_q)) != pure_q.index(max(pure_q)):
        mismatched_actions += 1

print(f"Checked {N} random states.")
print(f"Max abs Q-value difference: {max_abs_diff:.2e}")
print(f"Mismatched argmax actions: {mismatched_actions}/{N}")

assert max_abs_diff < 1e-3, "PureDQN diverges from torch DQN"
assert mismatched_actions == 0, "PureDQN picks a different best move than torch DQN on some states"
print("PASS: PureDQN is equivalent to the torch DQN for inference purposes.")
