"""Optional Stockfish UCI annotation. Not used by the inference/search pipeline."""

from pathlib import Path
import json

import chess
import chess.engine

from .data import STOCKFISH_SOURCE, write_jsonl


def read_fens(path: str | Path) -> list[str]:
    """One FEN or JSONL {fen: ...} per line; deduplicated in original order."""
    fens = []
    seen = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fen = json.loads(line)["fen"] if line.lstrip().startswith("{") else line.strip()
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("Invalid FEN: " + fen)
        if fen not in seen:
            fens.append(fen)
            seen.add(fen)
    return fens


def annotate(input_path: str | Path, output_path: str | Path,
             engine_path: str, depth: int = 8, threads: int = 1,
             hash_mb: int = 32) -> dict:
    if depth < 1 or threads < 1 or hash_mb < 1:
        raise ValueError("Invalid engine settings")
    fens = read_fens(input_path)
    if not fens:
        raise ValueError("No FEN positions supplied")
    rows = []
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        supported = engine.options
        config = {}
        if "Threads" in supported:
            config["Threads"] = threads
        if "Hash" in supported:
            config["Hash"] = hash_mb
        if config:
            engine.configure(config)
        identity = dict(engine.id)
        for fen in fens:
            board = chess.Board(fen)
            if board.is_game_over(claim_draw=True):
                continue
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            pov = info["score"].pov(board.turn)
            # Mate evaluations remain distinct; do NOT fabricate numerical CP labels.
            cp = pov.score()
            mate = pov.mate()
            rows.append({"fen": fen, "source": STOCKFISH_SOURCE,
                         "target_cp": cp, "mate_in": mate,
                         "bestmove_uci": info["pv"][0].uci() if info.get("pv") else None,
                         "pv_uci": [m.uci() for m in info.get("pv", [])],
                         "teacher": {"id": identity, "depth_limit": depth,
                                     "threads": threads, "hash_mb": hash_mb,
                                     "nodes": info.get("nodes"), "depth_reached": info.get("depth")}})
    write_jsonl(output_path, rows)
    return {"requested_positions": len(fens), "annotated_positions": len(rows),
            "mate_scores_have_null_target_cp": True,
            "output": str(output_path), "teacher_id": identity}
