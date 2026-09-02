"""FastAPI server exposing the trained DQN agent over HTTP.

Usage:
    alien-battle-serve                     # after `pip install dqn-alien-battle`
    uvicorn dqn_alien_battle.app:app --reload

POST a state vector to /predict-turn; interactive docs at /docs.
"""

from __future__ import annotations

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ._paths import default_model_path
from .model import ACTION_DIM, STATE_DIM, DQN

MODEL_PATH = default_model_path()

app = FastAPI(title="DQN Alien Battle Agent")

device = torch.device("cpu")
model = DQN(STATE_DIM, ACTION_DIM).to(device)
try:
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Could not find trained weights at '{MODEL_PATH}'. Train the agent first by "
        "running `alien-battle-train` (or `python -m dqn_alien_battle.train`), which "
        "writes a dqn_battle_agent.pth to the current directory."
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


def main() -> None:
    """Entry point for the `alien-battle-serve` console script."""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
