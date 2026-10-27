import chess

from chesslab.encoding import ENCODING_VERSION, N_PLANES, encode_board
from chesslab.material import material_cp


def test_encoding_shape_and_piece_count():
    board = chess.Board()
    x = encode_board(board)
    assert ENCODING_VERSION == "planes19-v1"
    assert len(x) == N_PLANES
    assert all(len(plane) == 8 and all(len(row) == 8 for row in plane) for plane in x)
    assert sum(sum(map(sum, plane)) for plane in x[:12]) == 32
    assert x[12][0][0] == 1
    assert all(x[p][0][0] == 1 for p in (13, 14, 15, 16))


def test_encoding_castle_en_passant_and_halfmove():
    board = chess.Board()
    board.push_san("e4")
    plane = encode_board(board)
    assert plane[12][0][0] == 0
    assert plane[17][2][4] == 1
    assert plane[18][0][0] == 0


def test_material_from_side_to_move():
    board = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
    assert material_cp(board) == 900
    board.turn = chess.BLACK
    assert material_cp(board) == -900
