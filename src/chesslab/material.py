"""Transparent non-neural baseline; centipawns from side-to-move perspective."""

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def material_cp(board: chess.Board) -> int:
    """Positive values favour the player to move; terminal rules are search's job."""
    total = sum(
        PIECE_VALUES[piece.piece_type] * (1 if piece.color == chess.WHITE else -1)
        for piece in board.piece_map().values()
    )
    return total if board.turn == chess.WHITE else -total
