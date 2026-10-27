"""Independent S004 partition, checkpoint, prediction and score audit."""
import json
from pathlib import Path
import math

import chess
import torch

from chesslab.data import read_jsonl, canonical_position
from chesslab.material import material_cp
from chesslab.model import load_evaluator
from chesslab.train import _tensors, _metrics
from s004_local import ROOT, RUN, SPLITS, SOURCES
from s002_local import sha, save


def main():
    plan=json.loads((RUN/'plan.json').read_text())
    frozen=json.loads((ROOT/'docs/experiments/S004-plan.json').read_text())
    assert plan==frozen
    progress=json.loads((RUN/'training.json').read_text())
    result=json.loads((RUN/'result.json').read_text())
    original={r['fen']:r for p in SOURCES for r in read_jsonl(p)}
    expected_hashes=json.loads((RUN/'split-hashes.json').read_text())
    for name,digest in expected_hashes.items():assert sha(SPLITS/name)==digest
    for path,digest in plan['development_sources'].items():assert sha(Path(path))==digest
    rows=[json.loads(line) for line in (RUN/'oof-predictions.jsonl').read_text().splitlines()]
    assert sha(RUN/'oof-predictions.jsonl')==result['oof_predictions_sha256']
    assert len(rows)==len({r['fen'] for r in rows})
    assert {r['fen'] for r in rows}=={fen for fen,r in original.items() if r['target_cp'] is not None}
    torch.set_num_threads(1)
    keys=set()
    outer_games=[]
    checkpoint_count=0
    max_prediction_error=0.
    for fold in plan['folds']:
        i=fold['fold']
        partitions={part:read_jsonl(SPLITS/f'fold-{i}-{part}.jsonl') for part in ['train','inner-val','outer']}
        game_sets={part:{r['game_id'] for r in subset} for part,subset in partitions.items()}
        for a,b in [('train','inner-val'),('train','outer'),('inner-val','outer')]:
            assert not game_sets[a]&game_sets[b]
            assert not ({canonical_position(r['fen']) for r in partitions[a]}&{canonical_position(r['fen']) for r in partitions[b]})
        assert game_sets['train']==set(fold['train_games'])
        assert game_sets['inner-val']==set(fold['inner_val_games'])
        assert game_sets['outer']==set(fold['outer_games'])
        outer_games.extend(game_sets['outer'])
        for subset in partitions.values():
            for row in subset:
                assert all(row[k]==v for k,v in original[row['fen']].items() if k!='split')
                assert row['development_origin_split']==original[row['fen']]['split']
        outer=[r for r in rows if r['fold']==i]
        x,_=_tensors(outer,normalized=True)
        for seed in plan['training_seeds']:
            trial=next(t for t in progress['trials'] if t['fold']==i and t['seed']==seed)
            checkpoint=Path(trial['checkpoint'])
            assert sha(checkpoint)==trial['checkpoint_sha256']
            model=load_evaluator(str(checkpoint))
            assert model.normalized and model.residual
            xv,yv=_tensors(partitions['inner-val'],normalized=True)
            assert _metrics(model,xv,yv)==trial['selected_validation']
            minimum=min([trial['initial_validation']['mae_pawns']]+[v['mae_pawns'] for v in trial['history']])
            assert trial['selected_validation']['mae_pawns']==minimum
            with torch.inference_mode(): predictions=model(x).tolist()
            for row,prediction in zip(outer,predictions):
                difference=abs(prediction-row['seed_predictions'][str(seed)])
                max_prediction_error=max(max_prediction_error,difference)
                assert difference<1e-6
            checkpoint_count+=1
    assert len(outer_games)==len(set(outer_games))==72
    for row in rows:
        assert chess.Board(row['fen']).is_valid()
        assert row['material_pawns']==material_cp(chess.Board(row['fen']))/100
    independent=[]
    for candidate in result['candidates']:
        scale=candidate['scale']
        games={}
        for row in rows:
            p=row['material_pawns']+scale*(sum(row['seed_predictions'].values())/2-row['material_pawns'])
            target=row['target_cp']/100
            game=games.setdefault(row['game_id'],{'fold':row['fold'],'deltas':[]})
            game['deltas'].append(abs(p-target)-abs(row['material_pawns']-target))
        means={g:sum(v['deltas'])/len(v['deltas']) for g,v in games.items()}
        delta=sum(means.values())/len(means)
        assert math.isclose(delta,candidate['mean_game_delta_pawns'],abs_tol=1e-10)
        independent.append({'scale':scale,'mean_game_delta_pawns':delta})
    selected=next(c for c in result['candidates'] if c['scale']==result['selected_scale'])
    passes=selected['scale']>0 and selected['mean_game_delta_pawns']<=-.05 and all(v<0 for v in selected['fold_deltas'].values()) and all(v<0 for v in selected['seed_deltas'].values()) and selected['game_bootstrap_95_interval_pawns'][1]<0
    assert result['gate_passed']==passes
    save(RUN/'audit.json',{'status':'PASS','development_games':72,'finite_oof_positions':len(rows),
                           'checkpoint_hashes_verified':checkpoint_count,'no_game_or_canonical_overlap_within_each_fold':True,
                           'each_source_game_outer_once':True,'inner_selected_metrics_verified':True,
                           'max_reloaded_prediction_difference':max_prediction_error,'independent_scores':independent,
                           'gate_independently_recomputed':passes,'fresh_test_required_only_after_gate_pass':True})
    print((RUN/'audit.json').read_text())

if __name__=='__main__':main()
