"""Tiny neural scalar evaluator. Trained independently, no engine at inference."""

import chess
import torch
from torch import nn

from .encoding import ENCODING_VERSION, N_PLANES, encode_board

MODEL_VERSION = "tinyconv-v1"


class TinyEvaluator(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(
            nn.Conv2d(N_PLANES, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Float32 output in pawn units from side-to-move perspective."""
        return self.net(positions).squeeze(-1)


def load_evaluator(checkpoint: str, device: str = "cpu") -> TinyEvaluator:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload["model_version"] != MODEL_VERSION or payload["encoding_version"] != ENCODING_VERSION:
        raise ValueError("Incompatible model / position encoding version")
    model = TinyEvaluator(width=payload["width"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


class NeuralCP:
    """Search-compatible callable. Explicitly keeps inference on CPU for simple UCI."""

    def __init__(self, checkpoint: str):
        self.model = load_evaluator(checkpoint)

    def __call__(self, board: chess.Board) -> int:
        features = torch.tensor([encode_board(board)], dtype=torch.float32)
        with torch.inference_mode():
            prediction_pawns = float(self.model(features).item())
        return int(round(100 * prediction_pawns))
