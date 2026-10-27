"""Deterministic small-scale supervised experiment, not a competitive engine."""

import hashlib
import json
import random
import copy
from pathlib import Path

import torch
from torch import nn

from .data import read_jsonl, canonical_position
from .encoding import ENCODING_VERSION, NORMALIZED_ENCODING_VERSION, encode_board
from .material import material_cp
from .model import MODEL_VERSION, RESIDUAL_MODEL_VERSION, TinyEvaluator


def _tensors(records: list[dict], normalized: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    import chess
    valid = [r for r in records if r.get("target_cp") is not None]
    if not valid:
        raise ValueError("Dataset has no finite centipawn targets (mate-only?)")
    features = torch.tensor(
        [encode_board(chess.Board(r["fen"]), normalized=normalized) for r in valid], dtype=torch.float32
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
          epochs: int = 2, seed: int = 7, width: int = 32, lr: float = 0.002,
          normalized: bool = False, residual: bool = False, loss: str = "mse",
          include_initial: bool = False) -> dict:
    if epochs < 1 or width < 1 or lr <= 0:
        raise ValueError("Invalid training parameters")
    if loss not in ("mse", "huber"):
        raise ValueError("Unknown loss")
    train_rows, val_rows = read_jsonl(train_path), read_jsonl(val_path)
    if not train_rows or not val_rows:
        raise ValueError("Train and validation sets must be nonempty")
    overlap = {canonical_position(r["fen"]) for r in train_rows} & {canonical_position(r["fen"]) for r in val_rows}
    if overlap:
        raise ValueError("Train/validation position overlap")
    games_train = {r["game_id"] for r in train_rows if "game_id" in r}
    games_val = {r["game_id"] for r in val_rows if "game_id" in r}
    if games_train & games_val:
        raise ValueError("Train/validation source game overlap")
    sources = sorted({r["source"] for r in train_rows + val_rows})
    if len(sources) != 1:
        raise ValueError("Do not mix synthetic and engine-labelled datasets in one run")
    if Path(out_path).exists():
        raise FileExistsError(out_path)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    xtrain, ytrain = _tensors(train_rows, normalized=normalized)
    xval, yval = _tensors(val_rows, normalized=normalized)
    model = TinyEvaluator(width, normalized=normalized, residual=residual)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if loss == "mse" else nn.HuberLoss(delta=1.0)
    initial = _metrics(model, xval, yval)
    history = []
    best_mae = initial["mae_pawns"] if include_initial else float("inf")
    best_state = copy.deepcopy(model.state_dict()) if include_initial else None
    best_epoch = 0 if include_initial else None
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
        if history[-1]["mae_pawns"] < best_mae:
            best_mae = history[-1]["mae_pawns"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = len(history)
    model.load_state_dict(best_state)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_version": RESIDUAL_MODEL_VERSION if normalized or residual else MODEL_VERSION,
                "encoding_version": NORMALIZED_ENCODING_VERSION if normalized else ENCODING_VERSION,
                "normalized": normalized, "residual": residual,
                "width": width, "state_dict": model.state_dict()}, out_path)
    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    result = {"experiment": "scalar-distillation",
              "status": "predictive evaluation only; no strength claim",
              "label_source": sources[0], "train_count": len(train_rows),
              "val_count": len(val_rows), "seed": seed, "epochs": epochs,
              "width": width, "parameters": sum(p.numel() for p in model.parameters()),
              "initial_validation": initial, "final_validation": history[-1],
              "history": history, "selected_epoch": best_epoch,
              "selected_validation": _metrics(model, xval, yval),
              "learning_rate": lr, "batch_size": 32, "device": "cpu",
              "normalized": normalized, "residual": residual, "loss": loss,
              "include_initial_in_selection": include_initial,
              "checkpoint_sha256": sha256}
    return result


def evaluate(checkpoint: str | Path, dataset_path: str | Path) -> dict:
    import chess
    from .model import load_evaluator
    rows = read_jsonl(dataset_path)
    if not rows:
        raise ValueError("Dataset empty")
    model = load_evaluator(str(checkpoint))
    x, y = _tensors(rows, normalized=model.normalized)
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
            "warning": "Value loss is not playing strength; mate targets excluded"}
