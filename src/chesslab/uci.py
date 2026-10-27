"""Minimal synchronous UCI research engine. Depth-only; not clock-safe yet."""

import argparse
import sys
from typing import TextIO

import chess

from .material import material_cp
from .search import AlphaBeta


def run_uci(input_stream: TextIO, output_stream: TextIO, evaluator=material_cp) -> None:
    board = chess.Board()
    searcher = AlphaBeta(evaluator)

    def send(line: str) -> None:
        print(line, file=output_stream, flush=True)

    for line in input_stream:
        parts = line.strip().split()
        if not parts:
            continue
        command = parts[0]
        if command == "uci":
            send("id name ChessIntelligenceLab S001")
            send("id author Chess Intelligence Lab")
            send("uciok")
        elif command == "isready":
            send("readyok")
        elif command == "ucinewgame":
            board = chess.Board()
        elif command == "position":
            try:
                if len(parts) < 2:
                    raise ValueError("position missing argument")
                if parts[1] == "startpos":
                    board = chess.Board()
                    move_start = 2
                elif parts[1] == "fen":
                    move_start = parts.index("moves") if "moves" in parts else len(parts)
                    board = chess.Board(" ".join(parts[2:move_start]))
                else:
                    raise ValueError("unknown position format")
                if move_start < len(parts) and parts[move_start] == "moves":
                    for move_uci in parts[move_start + 1:]:
                        board.push_uci(move_uci)
            except (ValueError, IndexError) as exc:
                print(f"UCI position error: {exc}", file=sys.stderr)
        elif command == "go":
            # S001 deliberately implements fixed-depth synchronous search only.
            depth = 1
            if "depth" in parts:
                try:
                    depth = max(1, int(parts[parts.index("depth") + 1]))
                except (ValueError, IndexError):
                    depth = 1
            outcome = searcher.best_move(board, depth=depth)
            send(f"info depth {outcome.depth} nodes {outcome.nodes} score cp {outcome.score_cp}")
            send("bestmove " + (outcome.move.uci() if outcome.move else "(none)"))
        elif command == "quit":
            break
        # stop, ponder, options and clock-aware time management: not implemented in S001.


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chess Intelligence Lab UCI research engine")
    parser.add_argument("--checkpoint", help="Trained neural evaluator checkpoint")
    args = parser.parse_args(argv)
    evaluator = material_cp
    if args.checkpoint:
        from .model import NeuralCP
        evaluator = NeuralCP(args.checkpoint)
    run_uci(sys.stdin, sys.stdout, evaluator)


if __name__ == "__main__":
    main()
