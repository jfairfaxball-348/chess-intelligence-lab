"""Game-grouped development folds and paired residual-correction diagnostics."""
from collections import defaultdict
import random

from .data import canonical_position


def grouped_folds(rows, seed=404):
    """Three outer folds; eight inner validation games, disjoint from outer games."""
    games=sorted({row['game_id'] for row in rows})
    if len(games)<18:
        raise ValueError('At least 18 source games required')
    keys=[canonical_position(row['fen']) for row in rows]
    if len(set(keys))!=len(keys):
        raise ValueError('Canonical duplicate positions in development data')
    random.Random(seed).shuffle(games)
    folds=[]
    for fold in range(3):
        outer=games[fold::3]
        remaining=[game for game in games if game not in outer]
        random.Random(seed+fold+1).shuffle(remaining)
        inner=remaining[:8]
        training=remaining[8:]
        folds.append({'fold':fold,'train_games':training,'inner_val_games':inner,'outer_games':outer})
    return folds


def _game_differences(rows,predictions):
    grouped=defaultdict(list)
    for row,prediction in zip(rows,predictions):
        target=row['target_cp']/100
        grouped[row['game_id']].append(abs(prediction-target)-abs(row['material_pawns']-target))
    return {game:sum(errors)/len(errors) for game,errors in grouped.items()}


def summarize_oof(rows,seeds=(7,19),scales=(0,.25,.5,1),bootstrap_draws=5000,bootstrap_seed=404):
    """Choose shrinkage using development OOF scores; this is not a final test."""
    if not rows or len(set(seeds))!=len(seeds):
        raise ValueError('Nonempty rows and distinct seeds required')
    if sorted({r['fold'] for r in rows})!=[0,1,2]:
        raise ValueError('Expected three outer folds')
    game_folds={}
    for row in rows:
        game=row['game_id']
        if game in game_folds and game_folds[game]!=row['fold']:
            raise ValueError('Source game crosses outer folds')
        game_folds[game]=row['fold']
    result=[]
    for scale in scales:
        predictions=[]
        seed_predictions={str(seed):[] for seed in seeds}
        for row in rows:
            material=row['material_pawns']
            correction=sum(row['seed_predictions'][str(seed)]-material for seed in seeds)/len(seeds)
            predictions.append(material+scale*correction)
            for seed in seeds:
                seed_predictions[str(seed)].append(material+scale*(row['seed_predictions'][str(seed)]-material))
        differences=_game_differences(rows,predictions)
        values=list(differences.values())
        delta=sum(values)/len(values)
        rng=random.Random(bootstrap_seed)
        bootstrap=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(bootstrap_draws))
        interval=[bootstrap[int(.025*bootstrap_draws)],bootstrap[min(bootstrap_draws-1,int(.975*bootstrap_draws))]]
        folds={str(fold):sum(v for game,v in differences.items() if game_folds[game]==fold)/sum(game_folds[g]==fold for g in differences) for fold in range(3)}
        per_seed={}
        for seed in seeds:
            differences_seed=_game_differences(rows,seed_predictions[str(seed)])
            per_seed[str(seed)]=sum(differences_seed.values())/len(differences_seed)
        passed=scale>0 and delta<=-.05 and all(v<0 for v in folds.values()) and all(v<0 for v in per_seed.values()) and interval[1]<0
        # Average per-game MAE, so long games do not outweigh short games.
        material_by_game=defaultdict(list)
        neural_by_game=defaultdict(list)
        for row,prediction in zip(rows,predictions):
            material_by_game[row['game_id']].append(abs(row['material_pawns']-row['target_cp']/100))
            neural_by_game[row['game_id']].append(abs(prediction-row['target_cp']/100))
        result.append({'scale':scale,'mean_game_delta_pawns':delta,'fold_deltas':folds,'seed_deltas':per_seed,
                       'game_bootstrap_95_interval_pawns':interval,'passes_gate':passed,
                       'mean_game_material_mae_pawns':sum(sum(e)/len(e) for e in material_by_game.values())/len(material_by_game),
                       'mean_game_neural_mae_pawns':sum(sum(e)/len(e) for e in neural_by_game.values())/len(neural_by_game),
                       'pooled_material_mae_pawns':sum(abs(r['material_pawns']-r['target_cp']/100) for r in rows)/len(rows),
                       'pooled_neural_mae_pawns':sum(abs(p-r['target_cp']/100) for r,p in zip(rows,predictions))/len(rows)})
    best=min(result,key=lambda candidate:(candidate['mean_game_neural_mae_pawns'],candidate['scale']))
    return {'positions':len(rows),'source_games':len(game_folds),'candidates':result,'selected_scale':best['scale'],
            'gate_passed':best['passes_gate'],'bootstrap_draws':bootstrap_draws,'bootstrap_seed':bootstrap_seed,
            'claim_limit':'Reused development data; scale selection is optimistic; fresh final test required after gate pass'}
