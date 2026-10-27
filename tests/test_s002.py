import json
import chess
import chess.engine
import pytest
from chesslab.teacher import annotate
from chesslab.data import canonical_position


def test_symmetry_and_clock_free_key():
    b = chess.Board()
    b.push_san('e4')
    c = b.copy()
    c.halfmove_clock = 19
    c.fullmove_number = 30
    assert canonical_position(b.fen()) == canonical_position(c.fen())
    assert canonical_position(b.fen()) == canonical_position(b.mirror().fen())
    b.castling_rights = 0
    assert canonical_position(b.fen()) == canonical_position(b.transform(chess.flip_horizontal).fen())


def test_annotation_recovery_and_config_guard(tmp_path, monkeypatch):
    engine_path = tmp_path/'engine.exe'
    engine_path.write_bytes(b'fake-engine-for-test')
    source, out = tmp_path/'input.txt', tmp_path/'labels.jsonl'
    b = chess.Board()
    fens = [b.fen()]
    b.push_san('e4')
    fens.append(b.fen())
    source.write_text('\n'.join(fens))
    calls = []
    class FakeEngine:
        options = {'Threads': None, 'Hash': None}
        id = {'name': 'fake'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def configure(self, settings): pass
        def analyse(self, board, limit, **kwargs):
            calls.append(board.fen())
            if len(calls) == 2:
                raise RuntimeError('simulated engine crash')
            return {'score': chess.engine.PovScore(chess.engine.Cp(23), board.turn),
                    'pv': [next(iter(board.legal_moves))], 'nodes': 42, 'depth': 8}
    monkeypatch.setattr(chess.engine.SimpleEngine, 'popen_uci', lambda path: FakeEngine())
    with pytest.raises(RuntimeError, match='simulated'):
        annotate(source,out,str(engine_path))
    prefix = out.read_bytes()
    assert len(prefix.splitlines()) == 1
    with out.open('ab') as stream: stream.write(b'{"partial"')
    result = annotate(source,out,str(engine_path))
    assert result['annotated_positions'] == 2
    assert out.read_bytes().startswith(prefix)
    assert calls == [fens[0],fens[1],fens[1]]
    before = out.read_bytes()
    annotate(source,out,str(engine_path))
    assert out.read_bytes() == before
    with pytest.raises(ValueError, match='configuration'):
        annotate(source,out,str(engine_path),depth=9)
