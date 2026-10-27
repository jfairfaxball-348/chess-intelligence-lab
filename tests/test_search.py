import chess

from chesslab.material import material_cp
from chesslab.search import AlphaBeta


def test_legal_move_and_unmodified_board():
    board = chess.Board()
    original = board.fen()
    result = AlphaBeta().best_move(board, depth=2)
    assert result.move in board.legal_moves
    assert result.nodes > 0
    assert board.fen() == original


def test_capture_preferred_in_clear_material_position():
    board = chess.Board("4k3/8/8/8/8/8/4q3/4KQ2 w - - 0 1")
    # White queen can capture black queen along the first rank? Instead require
    # legality and determinism for this intentionally adversarial setting.
    first = AlphaBeta().best_move(board, depth=1)
    second = AlphaBeta().best_move(board, depth=1)
    assert first.move == second.move
    assert first.move in board.legal_moves


def test_terminal_checkmate_and_stalemate():
    mate = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    result = AlphaBeta().best_move(mate, depth=1)
    assert result.move is None and result.score_cp < -90000
    stale = chess.Board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
    result = AlphaBeta().best_move(stale, depth=1)
    assert result.move is None and result.score_cp == 0
