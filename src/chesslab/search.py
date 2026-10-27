"""Independent depth-limited alpha-beta negamax; no tablebases or Stockfish calls."""

from collections.abc import Callable
from dataclasses import dataclass

import chess

from .material import material_cp

MATE_SCORE = 100_000
INFINITY = 1_000_000


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move | None
    score_cp: int
    nodes: int
    depth: int


class AlphaBeta:
    def __init__(self, evaluator: Callable[[chess.Board], int] = material_cp):
        self.evaluator = evaluator
        self.nodes = 0

    def _score(self, board: chess.Board, depth: int, alpha: int,
               beta: int, ply: int) -> int:
        self.nodes += 1
        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_game_over(claim_draw=True):
            return 0
        if depth == 0:
            return int(self.evaluator(board))
        best = -INFINITY
        for move in self._ordered(board):
            board.push(move)
            try:
                score = -self._score(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    @staticmethod
    def _ordered(board: chess.Board) -> list[chess.Move]:
        # Deterministic: captures first, otherwise UCI order.
        return sorted(board.legal_moves, key=lambda m: (not board.is_capture(m), m.uci()))

    def best_move(self, board: chess.Board, depth: int = 1) -> SearchResult:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.nodes = 0
        if board.is_game_over(claim_draw=True):
            return SearchResult(None, self._score(board, 0, -INFINITY, INFINITY, 0),
                                self.nodes, depth)
        best_move = None
        best = -INFINITY
        alpha, beta = -INFINITY, INFINITY
        for move in self._ordered(board):
            board.push(move)
            try:
                score = -self._score(board, depth - 1, -beta, -alpha, 1)
            finally:
                board.pop()
            if score > best:
                best, best_move = score, move
            alpha = max(alpha, score)
        return SearchResult(best_move, best, self.nodes, depth)
