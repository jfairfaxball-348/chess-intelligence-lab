"""Independent artifact audit; no model selection or test-set tuning."""
import json
from itertools import combinations
from pathlib import Path
import chess
import chess.pgn
from chesslab.data import read_jsonl, canonical_position
from s002_local import ROOT, RAW, SPLIT, TEACHER, RUN, sha, save

manifest = json.loads((SPLIT/'manifest.json').read_text())
keys, games, counts, stats = {}, {}, {}, {}
source_positions = {}
for path in RAW.glob('*.pgn'):
    with path.open() as stream: game = chess.pgn.read_game(stream)
    board = game.board()
    for ply, move in enumerate(game.mainline_moves(), 1):
        board.push(move)
        source_positions[(game.headers['GameId'], ply)] = board.fen()
for split in ('train','val','test'):
    rows = read_jsonl(TEACHER/f'{split}-provenance.jsonl')
    source = [json.loads(l) for l in (SPLIT/f'{split}.jsonl').read_text().splitlines()]
    assert len(rows) == len(source) == manifest[split]['positions']
    assert sha(SPLIT/f'{split}.jsonl') == manifest[split]['sha256']
    keys[split] = {canonical_position(r['fen']) for r in rows}
    games[split] = {r['game_id'] for r in rows}
    assert len(keys[split]) == len(rows)
    cp = []
    for row, origin in zip(rows,source):
        assert all(row[k] == v for k,v in origin.items())
        assert row['fen'] == source_positions[(row['game_id'], row['ply'])]
        board=chess.Board(row['fen'])
        assert board.is_valid() and not board.is_game_over(claim_draw=True)
        for uci in row['pv_uci']:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            board.push(move)
        assert (row['target_cp'] is None) == (row['mate_in'] is not None)
        assert row['teacher']['depth_reached'] >= 8
        if row['target_cp'] is not None: cp.append(row['target_cp'])
    counts[split] = len(rows)
    stats[split] = {'finite':len(cp),'min_cp':min(cp),'max_cp':max(cp),
                    'negative':sum(v < -100 for v in cp),'balanced':sum(-100<=v<=100 for v in cp),
                    'positive':sum(v>100 for v in cp)}
for a,b in combinations(keys,2):
    assert not keys[a]&keys[b]
    assert not games[a]&games[b]
for path in RAW.glob('*.pgn'):
    assert sha(path)==manifest['raw_hashes'][path.name]
    with path.open() as stream: game=chess.pgn.read_game(stream)
    assert not game.errors
    board=game.board()
    for move in game.mainline_moves():
        assert move in board.legal_moves
        board.push(move)
    assert game.headers['Result']==board.result(claim_draw=True)
save(RUN/'audit.json',{'status':'PASS','counts':counts,'finite_label_distribution':stats,
                      'cross_split_canonical_overlap':0,'cross_split_source_game_overlap':0,
                      'legal_source_games':80,'all_teacher_pvs_legal':True})
print((RUN/'audit.json').read_text())
