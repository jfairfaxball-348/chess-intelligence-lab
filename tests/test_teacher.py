import chess

from chesslab.teacher import read_fens


def test_read_fens_deduplicates_without_engine(tmp_path):
    p = tmp_path / "fens.txt"
    p.write_text(chess.STARTING_FEN + "\n" + chess.STARTING_FEN + "\n")
    assert read_fens(p) == [chess.STARTING_FEN]
