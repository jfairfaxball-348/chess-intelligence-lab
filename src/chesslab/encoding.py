"""Explicit, versioned position encoding with no tablebase or engine dependency."""

import chess

ENCODING_VERSION = "planes19-v1"
NORMALIZED_ENCODING_VERSION = "planes19-stm-v2"
N_PLANES = 19
_PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def encode_board(board: chess.Board, normalized: bool = False) -> list[list[list[float]]]:
    """19 x 8 x 8, indexed rank 1 to rank 8, file a to h.

    Planes 0..5: white PNBRQK, 6..11: black PNBRQK.
    12: white to move; 13..16: white/black K/Q castling rights;
    17: en-passant target square; 18: halfmove clock / 100 (clipped).
    With normalized=True, mirror black-to-move boards so the mover is white.
    Otherwise all planes preserve absolute board orientation.
    """
    if normalized and board.turn == chess.BLACK:
        board = board.mirror()
    planes = [[[0.0] * 8 for _ in range(8)] for _ in range(N_PLANES)]
    for square, piece in board.piece_map().items():
        offset = 0 if piece.color == chess.WHITE else 6
        idx = offset + _PIECES.index(piece.piece_type)
        planes[idx][chess.square_rank(square)][chess.square_file(square)] = 1.0
    flags = (
        board.turn == chess.WHITE,
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for plane_idx, active in enumerate(flags, start=12):
        if active:
            planes[plane_idx] = [[1.0] * 8 for _ in range(8)]
    if board.ep_square is not None:
        planes[17][chess.square_rank(board.ep_square)][chess.square_file(board.ep_square)] = 1.0
    halfmove = min(board.halfmove_clock, 100) / 100.0
    planes[18] = [[halfmove] * 8 for _ in range(8)]
    return planes
