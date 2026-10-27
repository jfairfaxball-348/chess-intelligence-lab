import random
import chess
import pytest
from chesslab.validation import grouped_folds, summarize_oof
from chesslab.data import synthetic_records


def test_grouped_folds_never_cross_source_games():
    rows=synthetic_records(36,404)
    for i,row in enumerate(rows):row['game_id']=f'game-{i//2:02d}'
    folds=grouped_folds(rows)
    assert folds==grouped_folds(rows)
    all_games={r['game_id'] for r in rows}
    outer=[]
    for fold in folds:
        t,v,o=(set(fold[key]) for key in ['train_games','inner_val_games','outer_games'])
        assert not (t&v or t&o or v&o)
        assert t|v|o==all_games
        assert len(v)==8
        outer.extend(o)
    assert set(outer)==all_games and len(outer)==len(all_games)
    with pytest.raises(ValueError,match='duplicate'):
        grouped_folds(rows+[rows[0]])


def _rows(prediction):
    return [{'game_id':f'game-{g}','fold':g%3,'target_cp':100,'material_pawns':0,
             'seed_predictions':{'7':prediction,'19':prediction}} for g in range(18)]


def test_gate_accepts_consistent_improvement_and_preserves_material():
    report=summarize_oof(_rows(1),bootstrap_draws=100)
    assert report['gate_passed'] and report['selected_scale']==1
    best=report['candidates'][-1]
    assert best['mean_game_delta_pawns']==-1
    assert best['game_bootstrap_95_interval_pawns']==[-1,-1]
    assert report['candidates'][0]['mean_game_delta_pawns']==0
    assert not report['candidates'][0]['passes_gate']


def test_gate_rejects_small_gains_or_one_bad_fold():
    assert not summarize_oof(_rows(.01),bootstrap_draws=100)['gate_passed']
    rows=_rows(1)
    for row in rows:
        if row['fold']==0:row['seed_predictions']={'7':-1,'19':-1}
    report=summarize_oof(rows,bootstrap_draws=100)
    assert not report['gate_passed']


def test_mean_game_metric_does_not_weight_long_games_more():
    rows=_rows(1)
    rows[0]['seed_predictions']={'7':-1,'19':-1}
    report=summarize_oof(rows+[dict(rows[0]) for _ in range(9)],scales=(1,),bootstrap_draws=100)
    assert report['candidates'][0]['mean_game_delta_pawns']==pytest.approx(-16/18)
    rows.append(dict(rows[0],fold=1))
    with pytest.raises(ValueError,match='crosses'):
        summarize_oof(rows,bootstrap_draws=100)
