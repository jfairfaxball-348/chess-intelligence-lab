from io import StringIO

from chesslab.uci import run_uci


def test_uci_handshake_and_bestmove():
    stdin = StringIO("uci\nisready\nposition startpos moves e2e4 e7e5\ngo depth 1\nquit\n")
    out = StringIO()
    run_uci(stdin, out)
    lines = out.getvalue().splitlines()
    assert "uciok" in lines and "readyok" in lines
    assert any(x.startswith("info depth 1 nodes ") for x in lines)
    assert any(x.startswith("bestmove ") and "(none)" not in x for x in lines)
