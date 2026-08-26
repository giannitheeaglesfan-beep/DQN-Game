from __future__ import annotations

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from model import ACTION_DIM, STATE_DIM, DQN

MODEL_PATH = "dqn_battle_agent.pth"

app = FastAPI(title="DQN Creature Battle Agent")

device = torch.device("cpu")
model = DQN(STATE_DIM, ACTION_DIM).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Could not find '{MODEL_PATH}'. Train the agent first by running "
        "`python train.py`, which produces this weights file."
    ) from exc
model.eval()


class PredictRequest(BaseModel):
    state: list[float] = Field(..., min_length=STATE_DIM, max_length=STATE_DIM)


class PredictResponse(BaseModel):
    chosen_move: int
    q_values: list[float]


@app.post("/predict-turn", response_model=PredictResponse)
def predict_turn(request: PredictRequest) -> PredictResponse:
    with torch.no_grad():
        state_t = torch.tensor(request.state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values_t = model(state_t)
        chosen_move = int(torch.argmax(q_values_t, dim=1).item())
        q_values = q_values_t.squeeze(0).tolist()

    return PredictResponse(chosen_move=chosen_move, q_values=q_values)
