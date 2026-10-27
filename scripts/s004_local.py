"""S004: nested, game-grouped residual-correction development gate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import chess
import torch

from chesslab.data import read_jsonl, write_jsonl
from chesslab.material import material_cp
from chesslab.model import load_evaluator
from chesslab.train import train, _tensors
from chesslab.validation import grouped_folds, summarize_oof
from s002_local import ROOT, sha, save

RUN=ROOT/'runs/s004'
SPLITS=ROOT/'data/splits/s004-cv-v001'
SOURCES=[ROOT/f'data/teacher/teacher-v001/{name}-provenance.jsonl' for name in ['train','val']]


def freeze_json(path,obj):
    if path.exists():
        assert json.loads(path.read_text())==obj,f'Immutable file differs: {path}'
    else: save(path,obj)


def prepare():
    rows=[row for path in SOURCES for row in read_jsonl(path)]
    folds=grouped_folds(rows)
    assert len({r['game_id'] for r in rows})==72
    config={'session':'S004','development_sources':{str(path):sha(path) for path in SOURCES},
            'development_games':72,'development_positions':len(rows),'fold_seed':404,'outer_folds':3,
            'inner_validation_games_per_fold':8,'training_seeds':[7,19], 'scales':[0,.25,.5,1],
            'model':'Existing normalized tinyconv-config-v2 material residual, zero correction initialization',
            'epochs':20,'learning_rate':.002,'width':32,'batch_size':32,'loss':'mse','cpu_threads':1,
            'epoch_selection':'Inner validation MAE including material initialization; outer labels not used for epochs',
            'scale_selection':'Lowest mean per-game OOF MAE of two-seed mean prediction; ties prefer smaller scale',
            'gate':{'minimum_mean_game_mae_improvement_pawns':.05,'all_three_outer_fold_deltas_negative':True,
                    'both_individual_seed_deltas_negative':True,'game_bootstrap_95_upper_below_zero':True,
                    'nonzero_selected_scale':True},
            'bootstrap_draws':5000,'bootstrap_seed':404,
            'conditional_final':'Only after PASS: 20 fresh games, seed 40361027+i, unchanged Stockfish generation/annotation settings; predict with frozen mean of six fold checkpoints at selected shrinkage; no extra training',
            'failure_pivot':'Stop CNN training and scaling. S005 should audit material-error mechanisms on existing development games before proposing a different model or source domain.',
            'limits':'Reused development corpus; no final-test claim from CV; no timed match or strength claim',
            'folds':folds}
    RUN.mkdir(parents=True,exist_ok=True)
    freeze_json(RUN/'plan.json',config)
    for fold in folds:
        for part,key in [('train','train_games'),('inner-val','inner_val_games'),('outer','outer_games')]:
            games=set(fold[key])
            subset=[dict(r,development_origin_split=r['split'],split=part) for r in rows if r['game_id'] in games]
            path=SPLITS/f"fold-{fold['fold']}-{part}.jsonl"
            if path.exists():
                assert read_jsonl(path)==subset
            else: write_jsonl(path,subset)
    freeze_json(RUN/'split-hashes.json',{p.name:sha(p) for p in sorted(SPLITS.glob('*.jsonl'))})
    print('Prepared 72 games, three nested folds, six bounded training runs',flush=True)
    return config


def fit(config):
    path=RUN/'training.json'
    report=json.loads(path.read_text()) if path.exists() else {
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'executed_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src/chesslab').glob('*.py'))},
        'runner_sha256':sha(Path(__file__)),'trials':[]}
    for fold in config['folds']:
        i=fold['fold']
        for seed in config['training_seeds']:
            checkpoint=ROOT/f'checkpoints/s004/fold-{i}-seed-{seed}.pt'
            old=next((t for t in report['trials'] if t['fold']==i and t['seed']==seed),None)
            if old:
                assert sha(checkpoint)==old['checkpoint_sha256']
                continue
            start=time.monotonic()
            metrics=train(SPLITS/f'fold-{i}-train.jsonl',SPLITS/f'fold-{i}-inner-val.jsonl',checkpoint,
                          epochs=20,seed=seed,width=32,lr=.002,normalized=True,residual=True,loss='mse',include_initial=True)
            report['trials'].append({'fold':i,'checkpoint':str(checkpoint),'seconds':time.monotonic()-start,**metrics})
            save(path,report)
            print('fold',i,'seed',seed,'epoch',metrics['selected_epoch'],'inner MAE',metrics['selected_validation']['mae_pawns'],flush=True)
    if 'finished_utc' not in report:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        save(path,report)
    return report


def summarize(config):
    report=json.loads((RUN/'training.json').read_text())
    assert len(report['trials'])==6 and 'finished_utc' in report
    rows=[]
    torch.set_num_threads(1)
    for i in range(3):
        source=[r for r in read_jsonl(SPLITS/f'fold-{i}-outer.jsonl') if r['target_cp'] is not None]
        x,_=_tensors(source,normalized=True)
        predictions={}
        for seed in config['training_seeds']:
            trial=next(t for t in report['trials'] if t['fold']==i and t['seed']==seed)
            assert sha(Path(trial['checkpoint']))==trial['checkpoint_sha256']
            model=load_evaluator(trial['checkpoint'])
            with torch.inference_mode(): prediction=model(x)
            assert torch.isfinite(prediction).all()
            predictions[str(seed)]=prediction.tolist()
        rows.extend([dict(r,fold=i,material_pawns=material_cp(chess.Board(r['fen']))/100,
                          seed_predictions={str(seed):predictions[str(seed)][j] for seed in config['training_seeds']})
                     for j,r in enumerate(source)])
    output=RUN/'oof-predictions.jsonl'
    if output.exists():
        assert [json.loads(l) for l in output.read_text().splitlines()]==rows
    else: write_jsonl(output,rows)
    result=summarize_oof(rows,seeds=tuple(config['training_seeds']),scales=tuple(config['scales']))
    result['oof_predictions_sha256']=sha(output)
    result['final_test_status']='PASS: proceed to preregistered fresh final' if result['gate_passed'] else 'NOT GENERATED: preregistered development gate failed'
    freeze_json(RUN/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','train','summarize'])
    args=parser.parse_args()
    config=prepare()
    if args.phase=='train':fit(config)
    elif args.phase=='summarize':summarize(config)

if __name__=='__main__':main()
