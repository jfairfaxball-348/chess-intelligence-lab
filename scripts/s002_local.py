"""Bounded local S002 pipeline. Re-run phases safely; large artifacts stay ignored."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import chess
import chess.engine
import chess.pgn

from chesslab.data import write_jsonl, read_jsonl, canonical_position
from chesslab.teacher import annotate

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/s002'
RAW = ROOT / 'data/raw/engine-v001'
SPLIT = ROOT / 'data/splits/engine-v001'
TEACHER = ROOT / 'data/teacher/teacher-v001'
ENGINE = ROOT / 'artifacts/stockfish/sf19/stockfish/stockfish-windows-x86-64-universal.exe'
SEED = 20261027


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')


canonical = canonical_position


def generate(games):
    RAW.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with chess.engine.SimpleEngine.popen_uci(str(ENGINE)) as engine:
        engine.configure({'Threads': 1, 'Hash': 32})
        for i in range(games):
            path = RAW / f'game-{i:04d}.pgn'
            if path.exists():
                continue
            rng = random.Random(SEED + i)
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers.update({'Event': 'S002 local engine generated', 'Site': 'local',
                                 'Date': '2026.10.27', 'Round': str(i),
                                 'White': 'Stockfish19 top3 seeded', 'Black': 'Stockfish19 top3 seeded',
                                 'GameId': f'engine-v001-{i:04d}', 'Seed': str(SEED+i)})
            node = game
            engine.configure({'Clear Hash': None})
            for ply in range(240):
                if board.is_game_over(claim_draw=True):
                    break
                infos = engine.analyse(board, chess.engine.Limit(depth=4), multipv=3, game=game)
                # Diverse reproducible engine-guided games, not representative human chess.
                choice = rng.choices(range(len(infos)), weights=[8, 2, 1][:len(infos)])[0]
                move = infos[choice]['pv'][0]
                board.push(move)
                node = node.add_variation(move)
            game.headers['Result'] = board.result(claim_draw=True)
            game.headers['Termination'] = 'normal' if board.is_game_over(claim_draw=True) else '240 ply cap'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(str(game) + '\n\n', encoding='utf-8')
            temporary.replace(path)
            print(f'game {i+1}/{games}; {board.ply()} plies; elapsed {time.monotonic()-started:.1f}s', flush=True)
    # Assign games before extraction; preserve exact assignment in manifest.
    ids = list(range(games))
    random.Random(SEED).shuffle(ids)
    nval = max(2, games//10)
    assignments = {'test': ids[:nval], 'val': ids[nval:2*nval], 'train': ids[2*nval:]}
    seen = set()
    manifest = {'seed': SEED, 'games': games, 'assignments': assignments,
                'source': 'Locally generated seeded Stockfish 19 top-3 depth-4 games; no third-party game corpus; Stockfish GPLv3',
                'source_caveat': 'Engine-guided shallow games with 240-ply cap; not representative of human chess',
                'dedup': 'Global clock-free transpositions, color/rank mirror, file mirror without castling; test then val then train',
                'sampling': 'Up to 40 positions per game, evenly stratified over available plies >= 8',
                'game_generation': {'depth': 4, 'multipv': 3, 'weights': [8,2,1], 'threads': 1, 'hash_mb': 32},
                'raw_hashes': {p.name: sha(p) for p in sorted(RAW.glob('*.pgn'))}}
    for partition, game_ids in assignments.items():
        records = []
        for i in game_ids:
            with (RAW / f'game-{i:04d}.pgn').open() as stream:
                game = chess.pgn.read_game(stream)
            assert not game.errors
            board = game.board()
            candidates = []
            for ply, move in enumerate(game.mainline_moves(), 1):
                board.push(move)
                if ply >= 8 and not board.is_game_over(claim_draw=True):
                    candidates.append((ply, board.fen()))
            for j in range(min(40, len(candidates))):
                ply, fen = candidates[(j * len(candidates)) // min(40, len(candidates))]
                key = canonical(fen)
                if key in seen:
                    continue
                seen.add(key)
                records.append({'fen': fen, 'game_id': f'engine-v001-{i:04d}', 'ply': ply, 'split': partition})
        path = SPLIT / f'{partition}.jsonl'
        if path.exists():
            previous = path.read_text()
            temporary = path.with_suffix('.tmp')
            write_jsonl(temporary, records)
            if previous != temporary.read_text():
                raise ValueError('Existing split differs; use a new version')
            temporary.unlink()
        else:
            write_jsonl(path, records)
        manifest[partition] = {'positions': len(records), 'sha256': sha(path), 'path': str(path)}
    save(SPLIT / 'manifest.json', manifest)
    return manifest


def label():
    TEACHER.mkdir(parents=True, exist_ok=True)
    results = {}
    for part in ('train', 'val', 'test'):
        started = time.monotonic()
        source = SPLIT / f'{part}.jsonl'
        target = TEACHER / f'{part}.jsonl'
        result = annotate(source, target, str(ENGINE), depth=8, threads=1, hash_mb=32)
        provenance = {r['fen']: r for r in [json.loads(l) for l in source.read_text().splitlines()]}
        # Keep append-only raw teacher file intact; enriched training file is immutable.
        enriched = TEACHER / f'{part}-provenance.jsonl'
        rows = [dict(r, **{k:v for k,v in provenance[r['fen']].items() if k != 'fen'}) for r in read_jsonl(target)]
        if not enriched.exists():
            write_jsonl(enriched, rows)
        results[part] = {**result, 'seconds': time.monotonic()-started, 'sha256': sha(enriched),
                         'path': str(enriched), 'mate_rows': sum(r['target_cp'] is None for r in rows),
                         'phase_counts': dict(Counter('opening' if r['ply']<=30 else 'endgame' if len(chess.Board(r['fen']).piece_map())<=10 else 'middlegame' for r in rows))}
        print(part, results[part], flush=True)
    save(RUN / 'annotation.json', results)


def fit():
    import torch
    from chesslab.train import train, evaluate, _tensors, _metrics
    from chesslab.model import TinyEvaluator
    paths = {p: TEACHER / f'{p}-provenance.jsonl' for p in ('train','val','test')}
    started = datetime.now(timezone.utc).isoformat()
    report = {'started_utc': started, 'git_sha': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
              'python': platform.python_version(), 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
              'engine_path': str(ENGINE), 'engine_sha256': sha(ENGINE), 'trials': []}
    # Predeclared validation-only choices. Test untouched until all selection is complete.
    for version, epochs, width, lr in [(1,20,32,.002),(2,30,32,.0005)]:
        checkpoint = ROOT / f'checkpoints/s002/model-v{version:03d}.pt'
        result = train(paths['train'], paths['val'], checkpoint, epochs=epochs, width=width, lr=lr, seed=7)
        report['trials'].append({'checkpoint':str(checkpoint), **result})
        save(RUN / 'training-progress.json', report)
        print('trial', version, result['selected_validation'], flush=True)
    best = min(report['trials'], key=lambda t:t['selected_validation']['mae_pawns'])
    report['selected_checkpoint'] = best['checkpoint']
    # Single final held-out evaluation of selected model and predeclared baselines.
    report['test'] = evaluate(best['checkpoint'], paths['test'])
    torch.manual_seed(7)
    x,y = _tensors(read_jsonl(paths['test']))
    report['test']['untrained'] = _metrics(TinyEvaluator(best['width']).eval(),x,y)
    report['test']['zero_baseline_mae_pawns'] = round(float(y.abs().mean()),6)
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['dataset_hashes'] = {p:sha(path) for p,path in paths.items()}
    save(RUN / 'result.json',report)
    print(json.dumps(report['test'],indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['generate','benchmark','label','train'])
    parser.add_argument('--games',type=int,default=80)
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if args.phase=='generate':
        print(json.dumps(generate(args.games),indent=2))
    elif args.phase=='benchmark':
        source=RUN/'benchmark-input.jsonl'
        if not source.exists():
            rows=[json.loads(l) for l in (SPLIT/'train.jsonl').read_text().splitlines()][:64]
            write_jsonl(source,rows)
        start=time.monotonic()
        result=annotate(source,RUN/'benchmark-labels.jsonl',str(ENGINE),depth=8)
        result['seconds']=time.monotonic()-start
        result['positions_per_second']=result['annotated_positions']/result['seconds']
        save(RUN/'benchmark.json',result)
        print(json.dumps(result,indent=2))
    elif args.phase=='label': label()
    elif args.phase=='train': fit()

if __name__=='__main__': main()
