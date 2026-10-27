import json

import chess
import pytest

from chesslab.data import generate_split, read_jsonl, synthetic_records


def test_legal_deterministic_data_and_split(tmp_path):
    assert synthetic_records(5, 19) == synthetic_records(5, 19)
    train, val = tmp_path / "t.jsonl", tmp_path / "v.jsonl"
    info = generate_split(train, val, 12, 6, 20)
    a, b = read_jsonl(train), read_jsonl(val)
    assert info["train_positions"] == 12
    assert len(b) == 6
    assert not {r["fen"] for r in a} & {r["fen"] for r in b}
    assert all(chess.Board(r["fen"]).is_valid() for r in a + b)
    assert {r["source"] for r in a + b} == {"synthetic_material_v1"}


def test_bad_source_is_rejected(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"fen": chess.STARTING_FEN, "source": "unverified",
                             "target_cp": 0}) + "\n")
    with pytest.raises(ValueError, match="source"):
        read_jsonl(p)
