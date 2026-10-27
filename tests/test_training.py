import json
import math

import chess
import pytest

torch = pytest.importorskip("torch")

from chesslab.data import generate_split
from chesslab.model import NeuralCP, load_evaluator
from chesslab.train import evaluate, train


def test_train_checkpoint_and_inference(tmp_path):
    t, v, ckpt = (tmp_path / s for s in ("train.jsonl", "val.jsonl", "model.pt"))
    generate_split(t, v, 12, 6, 3)
    metrics = train(t, v, ckpt, epochs=1, width=8)
    assert ckpt.exists()
    assert len(metrics["checkpoint_sha256"]) == 64
    net = load_evaluator(str(ckpt))
    assert net.width == 8
    score = NeuralCP(str(ckpt))(chess.Board())
    assert isinstance(score, int) and math.isfinite(score)
    result = evaluate(ckpt, v)
    assert result["positions"] == 6
    assert result["material_baseline_mae_pawns"] == 0.0


def test_reject_split_leakage(tmp_path):
    t, ckpt = tmp_path / "same.jsonl", tmp_path / "bad.pt"
    generate_split(t, tmp_path / "other.jsonl", 4, 2, 18)
    with pytest.raises(ValueError, match="overlap"):
        train(t, t, ckpt, epochs=1)
