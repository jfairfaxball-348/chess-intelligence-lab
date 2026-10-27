"""Verify that UCI reports a legal no-move result for terminal positions."""

from io import StringIO

from chesslab.uci import run_uci


def test_uci_terminal_mate_has_no_move():
    input_stream = StringIO(
        "uci\nposition fen 7k/6Q1/5K2/8/8/8/8/8 b - - 0 1\ngo depth 1\nquit\n"
    )
    output_stream = StringIO()
    run_uci(input_stream, output_stream)
    output = output_stream.getvalue().splitlines()
    assert "uciok" in output
    assert "bestmove (none)" in output
    assert any("score cp -100000" in line for line in output if line.startswith("info "))


def test_uci_terminal_stalemate_score_zero():
    input_stream = StringIO(
        "position fen 7k/5K2/6Q1/8/8/8/8/8 b - - 0 1\ngo depth 1\nquit\n"
    )
    output_stream = StringIO()
    run_uci(input_stream, output_stream)
    output = output_stream.getvalue().splitlines()
    assert "bestmove (none)" in output
    assert any("score cp 0" in line for line in output if line.startswith("info "))
