"""Frozen S003 diagnostics: eight short validation-only trials, one fresh test."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import random
import subprocess
import time

import chess
import chess.engine
import chess.pgn
import torch

from chesslab.data import canonical_position, read_jsonl, write_jsonl
from chesslab.material import material_cp
from chesslab.teacher import annotate
from chesslab.train import train, evaluate
from s002_local import ROOT, ENGINE, sha, save

RUN = ROOT/'runs/s003'
RAW = ROOT/'data/raw/s003-holdout-v001'
SPLITS = ROOT/'data/splits/s003-holdout-v001'
LABELS = ROOT/'data/teacher/s003-holdout-v001'
TRAIN = ROOT/'data/teacher/teacher-v001/train-provenance.jsonl'
VAL = ROOT/'data/teacher/teacher-v001/val-provenance.jsonl'
FRESH = LABELS/'test-provenance.jsonl'


def plan():
    config = {'session':'S003', 'base_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'factors':{'normalized':[False,True],'residual':[False,True],'loss':['mse','huber']},
              'epochs':20,'width':32,'seed':7,'learning_rate':.002,'batch_size':32,
              'selection':'Best validation MAE including epoch zero; between-trial validation MAE, stable trial order for ties',
              'decision':'Compare selected neural versus material and frozen S002 checkpoint on one fresh holdout; no further tuning afterward',
              'fresh_games':20,'fresh_generation_seed':30361027,'sampling':'40 evenly stratified positions/game at ply >=8',
              'generation':{'depth':4,'multipv':3,'weights':[8,2,1],'threads':1,'hash_mb':32,'ply_cap':240},
              'teacher':{'depth':8,'threads':1,'hash_mb':32,'engine_sha256':sha(ENGINE)},
              'train_sha256':sha(TRAIN),'val_sha256':sha(VAL),
              'residual_initialization':'Zero final-layer weights and bias, exact material at epoch zero',
              'limits':'Same CNN and parameter count; no timed match, no architecture expansion, one training seed',
              'source':'locally generated games, no third-party game corpus; Stockfish GPLv3'}
    path=RUN/'plan.json'
    if path.exists():
        old=json.loads(path.read_text())
        # A later Git commit does not change the frozen experiment configuration.
        config['base_git_sha']=old['base_git_sha']
        assert old==config,'Frozen plan differs'
    else: save(path,config)
    return config


def reserve(config):
    RAW.mkdir(parents=True,exist_ok=True)
    # Every new game belongs to the final test before position extraction.
    with chess.engine.SimpleEngine.popen_uci(str(ENGINE)) as engine:
        engine.configure({'Threads':1,'Hash':32})
        for index in range(config['fresh_games']):
            path=RAW/f'game-{index:04d}.pgn'
            if path.exists(): continue
            seed=config['fresh_generation_seed']+index
            rng=random.Random(seed)
            board=chess.Board()
            game=chess.pgn.Game()
            game.headers.update({'Event':'S003 reserved final holdout','Site':'local','Date':'2026.10.27',
                                 'White':'Stockfish19 top3 seeded','Black':'Stockfish19 top3 seeded',
                                 'Round':str(index),'GameId':f's003-holdout-v001-{index:04d}', 'Seed':str(seed)})
            node=game
            engine.configure({'Clear Hash':None})
            for ply in range(240):
                if board.is_game_over(claim_draw=True): break
                infos=engine.analyse(board,chess.engine.Limit(depth=4),multipv=3,game=game)
                choice=rng.choices(range(len(infos)),weights=[8,2,1][:len(infos)])[0]
                move=infos[choice]['pv'][0]
                board.push(move)
                node=node.add_variation(move)
            game.headers['Result']=board.result(claim_draw=True)
            game.headers['Termination']='normal' if board.is_game_over(claim_draw=True) else '240 ply cap'
            temp=path.with_suffix('.tmp')
            temp.write_text(str(game)+'\n\n',encoding='utf-8')
            temp.replace(path)
    seen=set()
    # Include old test for duplicate exclusion only, without inspecting targets or metrics.
    for split in ('train','val','test'):
        for line in (ROOT/f'data/splits/engine-v001/{split}.jsonl').read_text().splitlines():
            seen.add(canonical_position(json.loads(line)['fen']))
    rows=[]
    for path in sorted(RAW.glob('*.pgn')):
        with path.open() as stream: game=chess.pgn.read_game(stream)
        assert not game.errors
        board=game.board()
        candidates=[]
        for ply,move in enumerate(game.mainline_moves(),1):
            board.push(move)
            if ply>=8 and not board.is_game_over(claim_draw=True): candidates.append((ply,board.fen()))
        count=min(40,len(candidates))
        for j in range(count):
            ply,fen=candidates[j*len(candidates)//count]
            key=canonical_position(fen)
            if key in seen: continue
            seen.add(key)
            rows.append({'fen':fen,'game_id':game.headers['GameId'],'ply':ply,'split':'test'})
    SPLITS.mkdir(parents=True,exist_ok=True)
    path=SPLITS/'test.jsonl'
    if path.exists():
        assert [json.loads(l) for l in path.read_text().splitlines()]==rows
    else: write_jsonl(path,rows)
    save(SPLITS/'manifest.json',{'partition':'final test only','positions':len(rows), 'games':20,
                               'seed':config['fresh_generation_seed'], 'split_sha256':sha(path),
                               'canonical_exclusion':'All S002 splits plus within fresh holdout',
                               'raw_hashes':{p.name:sha(p) for p in sorted(RAW.glob('*.pgn'))}})
    print('Reserved fresh holdout positions:',len(rows),flush=True)


def diagnostics():
    result={}
    for split,path in [('train',TRAIN),('val',VAL)]:
        rows=[r for r in read_jsonl(path) if r['target_cp'] is not None]
        error=[(r,abs(material_cp(chess.Board(r['fen']))-r['target_cp'])/100) for r in rows]
        result[split]={'s002':evaluate(ROOT/'checkpoints/s002/model-v001.pt',path),'material_by_turn':{}}
        for color in (True,False):
            errors=[e for r,e in error if chess.Board(r['fen']).turn==color]
            result[split]['material_by_turn']['white' if color else 'black']={'n':len(errors),'mae_pawns':sum(errors)/len(errors)}
    save(RUN/'diagnostics.json',result)


def fit(config):
    progress_path=RUN/'training.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'executed_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'numerical_source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src/chesslab').glob('*.py'))},
        'runner_sha256':sha(Path(__file__)), 'trials':[]}
    for index,(normalized,residual,loss) in enumerate(itertools.product([False,True],[False,True],['mse','huber']),1):
        checkpoint=ROOT/f'checkpoints/s003/model-v{index:03d}.pt'
        old=next((t for t in progress['trials'] if t['index']==index),None)
        if old:
            assert sha(checkpoint)==old['checkpoint_sha256']
            continue
        start=time.monotonic()
        metrics=train(TRAIN,VAL,checkpoint,epochs=config['epochs'],width=32,seed=7,lr=.002,
                      normalized=normalized,residual=residual,loss=loss,include_initial=True)
        progress['trials'].append({'index':index,'checkpoint':str(checkpoint),'seconds':time.monotonic()-start,**metrics})
        save(progress_path,progress)
        print(index,normalized,residual,loss,metrics['selected_validation'],flush=True)
    selected=min(progress['trials'],key=lambda t:t['selected_validation']['mae_pawns'])
    progress['selected_index']=selected['index']
    progress['selected_checkpoint']=selected['checkpoint']
    progress['selection_frozen_utc']=datetime.now(timezone.utc).isoformat()
    save(progress_path,progress)
    return progress


def final(config,progress):
    if (RUN/'result.json').exists(): raise FileExistsError('Final result already exists; do not retune/re-evaluate')
    LABELS.mkdir(parents=True,exist_ok=True)
    raw=LABELS/'test.jsonl'
    annotation=annotate(SPLITS/'test.jsonl',raw,str(ENGINE),depth=8,threads=1,hash_mb=32)
    origins={r['fen']:r for r in [json.loads(l) for l in (SPLITS/'test.jsonl').read_text().splitlines()]}
    if not FRESH.exists():
        write_jsonl(FRESH,[dict(r,**{k:v for k,v in origins[r['fen']].items() if k!='fen'}) for r in read_jsonl(raw)])
    selected=progress['selected_checkpoint']
    torch.set_num_threads(1)
    result={'plan':config,'selection':progress,'fresh_test':evaluate(selected,FRESH),
            'frozen_s002_on_fresh_test':evaluate(ROOT/'checkpoints/s002/model-v001.pt',FRESH),
            'fresh_test_sha256':sha(FRESH),'annotation':annotation,
            'finished_utc':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,
            'model_version':'tinyconv-config-v2','encoding':'absolute or STM according to checkpoint',
            'claim_limit':'Predictive accuracy only; correlated positions; no playing strength claim'}
    save(RUN/'result.json',result)
    print(json.dumps(result['fresh_test'],indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','train','final'])
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    config=plan()
    if args.phase=='prepare': reserve(config);diagnostics()
    elif args.phase=='train': fit(config)
    else: final(config,json.loads((RUN/'training.json').read_text()))

if __name__=='__main__': main()
