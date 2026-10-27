"""Deterministic small-scale supervised experiment, not a competitive engine."""

import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn

from .data import read_jsonl
from .encoding import ENCODING_VERSION, encode_board
from .material import material_cp
from .model import MODEL_VERSION, TinyEvaluator


def _tensors(records: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    import chess
    valid = [r for r in records if r.get("target_cp") is not None]
    if not valid:
        raise ValueError("Dataset has no finite centipawn targets (mate-only?)")
    features = torch.tensor(
        [encode_board(chess.Board(r["fen"])) for r in valid], dtype=torch.float32
    )
    targets = torch.tensor([float(r["target_cp"]) / 100.0 for r in valid],
                           dtype=torch.float32)
    if not torch.isfinite(targets).all():
        raise ValueError("Nonfinite target")
    return features, targets


def _metrics(model: TinyEvaluator, positions: torch.Tensor, targets: torch.Tensor) -> dict:
    with torch.inference_mode():
        predictions = model(positions)
        errors = (predictions - targets).abs()
        return {"mae_pawns": round(float(errors.mean()), 6),
                "rmse_pawns": round(float(torch.sqrt(((predictions - targets) ** 2).mean())), 6)}


def train(train_path: str | Path, val_path: str | Path, out_path: str | Path,
          epochs: int = 2, seed: int = 7, width: int = 32, lr: float = 0.002) -> dict:
    if epochs < 1 or width < 1 or lr <= 0:
        raise ValueError("Invalid training parameters")
    train_rows, val_rows = read_jsonl(train_path), read_jsonl(val_path)
    if not train_rows or not val_rows:
        raise ValueError("Train and validation sets must be nonempty")
    overlap = {r["fen"] for r in train_rows} & {r["fen"] for r in val_rows}
    if overlap:
        raise ValueError("Train/validation position overlap")
    sources = sorted({r["source"] for r in train_rows + val_rows})
    if len(sources) != 1:
        raise ValueError("Do not mix synthetic and engine-labelled datasets in one run")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    xtrain, ytrain = _tensors(train_rows)
    xval, yval = _tensors(val_rows)
    model = TinyEvaluator(width)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    initial = _metrics(model, xval, yval)
    history = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xtrain))
        for ids in order.split(32):
            optimiser.zero_grad()
            loss = loss_fn(model(xtrain[ids]), ytrain[ids])
            loss.backward()
            optimiser.step()
        model.eval()
        history.append(_metrics(model, xval, yval))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_version": MODEL_VERSION, "encoding_version": ENCODING_VERSION,
                "width": width, "state_dict": model.state_dict()}, out_path)
    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    result = {"experiment": "S001-synthetic-distillation-smoke",
              "status": "synthetic label demonstration; no strength claim",
              "label_source": sources[0], "train_count": len(train_rows),
              "val_count": len(val_rows), "seed": seed, "epochs": epochs,
              "width": width, "parameters": sum(p.numel() for p in model.parameters()),
              "initial_validation": initial, "final_validation": history[-1],
              "history": history, "checkpoint_sha256": sha256}
    return result


def evaluate(checkpoint: str | Path, dataset_path: str | Path) -> dict:
    import chess
    from .model import load_evaluator
    rows = read_jsonl(dataset_path)
    if not rows:
        raise ValueError("Dataset empty")
    x, y = _tensors(rows)
    model = load_evaluator(str(checkpoint))
    model.eval()
    net = _metrics(model, x, y)
    baseline = torch.tensor(
        [material_cp(chess.Board(row["fen"])) / 100.0
         for row in rows if row.get("target_cp") is not None], dtype=torch.float32
    )
    baseline_mae = float((baseline - y).abs().mean())
    return {"label_sources": sorted({r["source"] for r in rows}),
            "positions": len(y), "neural": net,
            "material_baseline_mae_pawns": round(baseline_mae, 6),
            "warning": "Value loss is not playing strength; synthetic labels copy material baseline"}
