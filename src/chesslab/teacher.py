"""Optional Stockfish UCI annotation. Not used by the inference/search pipeline."""

from pathlib import Path
import json
import hashlib
import os

import chess
import chess.engine

from .data import STOCKFISH_SOURCE


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
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    engine_sha = hashlib.sha256(Path(engine_path).read_bytes()).hexdigest()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    settings = {"input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                "engine_sha256": engine_sha, "depth": depth, "threads": threads,
                "hash_mb": hash_mb, "reset_hash_per_position": True}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != settings:
            raise ValueError("Resume configuration mismatch")
    elif output.exists():
        raise ValueError("Existing output without resume manifest")
    else:
        manifest_path.write_text(json.dumps(settings, indent=2) + "\n")
    # Only an incomplete final line can be repaired; complete malformed rows fail closed.
    rows = []
    if output.exists():
        raw = output.read_bytes()
        end = raw.rfind(b"\n") + 1
        if end != len(raw):
            with output.open("r+b") as stream:
                stream.truncate(end)
        rows = [json.loads(line) for line in raw[:end].splitlines()]
    eligible = [fen for fen in fens if not chess.Board(fen).is_game_over(claim_draw=True)]
    if [row["fen"] for row in rows] != eligible[:len(rows)]:
        raise ValueError("Resume output is not an input prefix")
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
        for fen in eligible[len(rows):]:
            board = chess.Board(fen)
            if board.is_game_over(claim_draw=True):
                continue
            engine.configure({"Clear Hash": None})
            info = engine.analyse(board, chess.engine.Limit(depth=depth), game=object())
            pov = info["score"].pov(board.turn)
            # Mate evaluations remain distinct; do NOT fabricate numerical CP labels.
            cp = pov.score()
            mate = pov.mate()
            row = {"fen": fen, "source": STOCKFISH_SOURCE,
                         "target_cp": cp, "mate_in": mate,
                         "bestmove_uci": info["pv"][0].uci() if info.get("pv") else None,
                         "pv_uci": [m.uci() for m in info.get("pv", [])],
                         "teacher": {"id": identity, "depth_limit": depth,
                                     "threads": threads, "hash_mb": hash_mb,
                                     "engine_sha256": engine_sha,
                                     "nodes": info.get("nodes"), "depth_reached": info.get("depth")}}
            with output.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(row)
    return {"requested_positions": len(fens), "annotated_positions": len(rows),
            "mate_scores_have_null_target_cp": True,
            "output": str(output_path), "teacher_id": identity}
