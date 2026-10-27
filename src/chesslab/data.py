"""Reproducible tiny synthetic datasets. Never mislabel these as Stockfish outputs."""

import json
import random
from pathlib import Path
from typing import Iterable

import chess

from .material import material_cp

SYNTHETIC_SOURCE = "synthetic_material_v1"
STOCKFISH_SOURCE = "stockfish_uci"


def canonical_position(fen: str) -> str:
    """Clock-free transpositions and orthodox color/rank and safe file symmetries."""
    board = chess.Board(fen)
    boards = [board, board.mirror()]
    if not board.castling_rights:
        boards += [b.transform(chess.flip_horizontal) for b in list(boards)]
    return min(" ".join(b.fen(en_passant="legal").split()[:4]) for b in boards)


def random_position(rng: random.Random) -> chess.Board:
    board = chess.Board()
    for _ in range(rng.randint(4, 70)):
        if board.is_game_over(claim_draw=True):
            break
        board.push(rng.choice(tuple(board.legal_moves)))
    return board


def synthetic_records(count: int, seed: int, excluded: Iterable[str] = ()) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    seen = set(excluded)
    rng = random.Random(seed)
    records = []
    for _ in range(max(1000, count * 100)):
        board = random_position(rng)
        fen = board.fen()
        if fen in seen or board.is_game_over(claim_draw=True):
            continue
        seen.add(fen)
        records.append({"fen": fen, "target_cp": material_cp(board), "source": SYNTHETIC_SOURCE})
        if len(records) == count:
            return records
    raise RuntimeError("Unable to generate enough unique legal nonterminal positions")


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for record in records:
            out.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or "fen" not in item or "source" not in item:
                raise ValueError(f"invalid record on line {i}")
            board = chess.Board(item["fen"])
            if not board.is_valid():
                raise ValueError(f"invalid board position on line {i}")
            if item["source"] not in (SYNTHETIC_SOURCE, STOCKFISH_SOURCE):
                raise ValueError(f"unrecognised label source on line {i}")
            records.append(item)
    return records


def generate_split(train_path: str | Path, val_path: str | Path,
                   n_train: int, n_val: int, seed: int) -> dict:
    train = synthetic_records(n_train, seed)
    val = synthetic_records(n_val, seed + 1, excluded=(r["fen"] for r in train))
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    assert not ({r["fen"] for r in train} & {r["fen"] for r in val})
    return {"label_source": SYNTHETIC_SOURCE, "seed": seed,
            "train_positions": len(train), "val_positions": len(val),
            "split_policy": "independent RNG seeds; cross-split exact-FEN exclusion"}
