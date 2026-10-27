"""Audit frozen S003 artifacts and report paired, game-resampled uncertainty."""
import json
from pathlib import Path
import random

import chess
import chess.pgn
import torch

from chesslab.data import read_jsonl, canonical_position
from chesslab.material import material_cp
from chesslab.model import load_evaluator
from chesslab.train import _tensors
from s003_local import ROOT, RUN, RAW, SPLITS, LABELS, TRAIN, VAL, FRESH
from s002_local import sha, save


def main():
    result=json.loads((RUN/'result.json').read_text())
    origins=[json.loads(l) for l in (SPLITS/'test.jsonl').read_text().splitlines()]
    rows=read_jsonl(FRESH)
    assert len(rows)==len(origins)
    assert sha(FRESH)==result['fresh_test_sha256']
    known=set()
    old_ids=set()
    for split in ('train','val','test'):
        old=[json.loads(l) for l in (ROOT/f'data/splits/engine-v001/{split}.jsonl').read_text().splitlines()]
        known.update(canonical_position(r['fen']) for r in old)
        old_ids.update(r['game_id'] for r in old)
    positions={}
    trajectories=set()
    for folder in (ROOT/'data/raw/engine-v001',RAW):
        for path in sorted(folder.glob('*.pgn')):
            with path.open() as stream: game=chess.pgn.read_game(stream)
            assert not game.errors
            moves=tuple(m.uci() for m in game.mainline_moves())
            assert moves not in trajectories,'Duplicate full source game trajectory'
            trajectories.add(moves)
            board=game.board()
            for ply,move in enumerate(game.mainline_moves(),1):
                assert move in board.legal_moves
                board.push(move)
                if folder==RAW: positions[(game.headers['GameId'],ply)]=board.fen()
            assert game.headers['Result']==board.result(claim_draw=True)
    manifest=json.loads((SPLITS/'manifest.json').read_text())
    for path in RAW.glob('*.pgn'): assert sha(path)==manifest['raw_hashes'][path.name]
    fresh_keys=set()
    for row,origin in zip(rows,origins):
        assert all(row[k]==v for k,v in origin.items())
        assert row['game_id'] not in old_ids
        assert row['fen']==positions[(row['game_id'],row['ply'])]
        key=canonical_position(row['fen'])
        assert key not in known and key not in fresh_keys
        fresh_keys.add(key)
        board=chess.Board(row['fen'])
        assert board.is_valid() and not board.is_game_over(claim_draw=True)
        assert (row['target_cp'] is None)==(row['mate_in'] is not None)
        assert row['teacher']['engine_sha256']==result['plan']['teacher']['engine_sha256']
        assert row['teacher']['depth_reached']>=8
        for move in row['pv_uci']:
            assert chess.Move.from_uci(move) in board.legal_moves
            board.push_uci(move)
    for trial in result['selection']['trials']:
        assert sha(Path(trial['checkpoint']))==trial['checkpoint_sha256']
    model=load_evaluator(result['selection']['selected_checkpoint'])
    torch.set_num_threads(1)
    valid=[r for r in rows if r['target_cp'] is not None]
    x,y=_tensors(valid,normalized=model.normalized)
    with torch.inference_mode():
        neural=(model(x)-y).abs().tolist()
    grouped={}
    for r,error in zip(valid,neural):
        material_error=abs(material_cp(chess.Board(r['fen']))-r['target_cp'])/100
        group=grouped.setdefault(r['game_id'],{'n':0,'neural_sum':0.,'material_sum':0.})
        group['n']+=1
        group['neural_sum']+=error
        group['material_sum']+=material_error
    pooled_delta=sum(g['neural_sum']-g['material_sum'] for g in grouped.values())/len(valid)
    rng=random.Random(303)
    groups=list(grouped.values())
    samples=[]
    for _ in range(5000):
        draw=rng.choices(groups,k=len(groups))
        samples.append(sum(g['neural_sum']-g['material_sum'] for g in draw)/sum(g['n'] for g in draw))
    samples.sort()
    save(RUN/'audit.json',{'status':'PASS','fresh_positions':len(rows),'finite_positions':len(valid),
                           'mate_positions':len(rows)-len(valid),'fresh_source_games':20,'finite_source_games':len(grouped),
                           'canonical_overlap_with_all_s002_splits':0,'duplicate_full_game_trajectories':0,
                           'all_source_fens_and_teacher_pvs_legal':True,
                           'checkpoint_hashes_verified':len(result['selection']['trials']),
                           'paired_pooled_neural_minus_material_mae_pawns':pooled_delta,
                           'game_bootstrap_95_percentile_interval_pawns':[samples[125],samples[4874]],
                           'bootstrap_replicates':5000,'bootstrap_seed':303,
                           'uncertainty_limit':'Game resampling of this shallow-engine corpus; not generalization to human chess or strength',
                           'per_game':grouped})
    print((RUN/'audit.json').read_text())

if __name__=='__main__':main()
