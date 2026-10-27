"""Tiny neural scalar evaluator. Trained independently, no engine at inference."""

import chess
import torch
from torch import nn

from .encoding import ENCODING_VERSION, NORMALIZED_ENCODING_VERSION, N_PLANES, encode_board

MODEL_VERSION = "tinyconv-v1"
RESIDUAL_MODEL_VERSION = "tinyconv-config-v2"


class TinyEvaluator(nn.Module):
    def __init__(self, width: int = 32, normalized: bool = False, residual: bool = False):
        super().__init__()
        self.width = width
        self.normalized = normalized
        self.residual = residual
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
        if residual:
            # Begin at the material baseline; learn only a correction to it.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Float32 output in pawn units from side-to-move perspective."""
        prediction = self.net(positions).squeeze(-1)
        if self.residual:
            values = positions.new_tensor([1, 3.2, 3.3, 5, 9, 0])
            pieces = positions[:, :12].sum(dim=(2, 3))
            white_material = ((pieces[:, :6] - pieces[:, 6:]) * values).sum(dim=1)
            stm_sign = 2 * positions[:, 12, 0, 0] - 1
            prediction = prediction + white_material * stm_sign
        return prediction


def load_evaluator(checkpoint: str, device: str = "cpu") -> TinyEvaluator:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    normalized = payload.get("normalized", False)
    residual = payload.get("residual", False)
    expected_encoding = NORMALIZED_ENCODING_VERSION if normalized else ENCODING_VERSION
    expected_model = RESIDUAL_MODEL_VERSION if normalized or residual else MODEL_VERSION
    if payload["model_version"] != expected_model or payload["encoding_version"] != expected_encoding:
        raise ValueError("Incompatible model / position encoding version")
    model = TinyEvaluator(width=payload["width"], normalized=normalized, residual=residual).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


class NeuralCP:
    """Search-compatible callable. Explicitly keeps inference on CPU for simple UCI."""

    def __init__(self, checkpoint: str):
        self.model = load_evaluator(checkpoint)

    def __call__(self, board: chess.Board) -> int:
        features = torch.tensor([encode_board(board, normalized=self.model.normalized)], dtype=torch.float32)
        with torch.inference_mode():
            prediction_pawns = float(self.model(features).item())
        return int(round(100 * prediction_pawns))
