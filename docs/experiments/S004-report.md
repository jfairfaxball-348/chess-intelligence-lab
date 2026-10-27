# S004 - grouped development validation; gate failed

S004 completed its bounded local experiment on 27 October 2026. Conservative
residual correction produced small development improvements, but no candidate
met the preregistered meaningful-progress gate. Training stopped as planned.
No new source games or final test corpus were generated. Material remains the
working predictive baseline; playing strength is unmeasured.

## Fixed comparison and result

Only the existing S002 training and validation corpus was used: 72 source games,
2,836 positions, 2,683 finite CP labels and 153 explicit mate labels. The latter
remain excluded from scalar metrics. S002's old test and S003's later test were
not included in S004 development folds. This is reused development data, not an
unseen confirmation set.

Games were shuffled with seed 404 and divided into three 24-game outer folds.
For each outer fold, the remaining 48 games were shuffled deterministically:
40 for training, eight for inner-validation epoch selection. Both training seeds
(7 and 19) use the same per-fold assignments. Every source game is in an outer
fold exactly once. Existing global canonical deduplication is preserved, with no
source-game or canonical-position overlap among the three roles within a fold.

Six models used the existing normalized material-residual CNN: width 32, 15,841
parameters, MSE/AdamW, learning rate 0.002, batch 32, 20 epochs, one CPU thread.
The final correction layer starts at zero. Epoch selection uses inner validation
MAE including epoch zero. The selected epochs were:

| Outer fold | Seed 7 | Seed 19 |
| --- | ---: | ---: |
| 0 | 13 | 9 |
| 1 | 16 | 15 |
| 2 | 5 | 4 |

For each outer position, average the two seed predictions and shrink the learned
correction using `prediction = material + alpha * (seed_mean - material)`.
The four preregistered alpha values were 0, 0.25, 0.5 and 1. Alpha zero is exactly
the material control. Each model is evaluated on outer games absent from both
its training and epoch-selection data.

Selection minimizes mean per-game out-of-fold MAE, with ties preferring smaller
alpha. Every game has equal weight regardless of its sampled position count.

| Correction scale | Mean game MAE | Difference from material | Game-bootstrap 95% interval for difference |
| --- | ---: | ---: | --- |
| 0, material control | 2.116065 | 0 | [0, 0] |
| 0.25 | 2.102750 | -0.013316 | [-0.021851, -0.005282] |
| 0.5, selected | 2.100821 | -0.015244 | [-0.032033, +0.001182] |
| 1 | 2.123295 | +0.007230 | [-0.026536, +0.041864] |

Negative differences favor the correction. Intervals resample whole source games,
5,000 draws with seed 404. They are descriptive development intervals, without
selection or multiple-comparison adjustment; they do not establish performance
on a new corpus or chess-playing strength. Folds share training data, so this
bootstrap also does not capture retraining uncertainty or independent replicate
experiments. The two seeds provide a limited check of training variability.

## Preregistered decision

Before training, the plan required the selected nonzero correction to:

- improve mean game MAE by at least 0.05 pawns;
- improve all three outer folds;
- improve each training seed in aggregate;
- have its game-bootstrap 95% upper endpoint below zero.

Alpha 0.5 was selected. Its fold differences were +0.003771, -0.032904 and
-0.016601 pawns. Its individual-seed differences were -0.009956 and -0.015652.
It misses the 0.05-pawn threshold, worsens fold 0 and has an interval containing
zero. The gate therefore failed. Alpha 0.25 improved every fold and both seeds,
but its 0.013316-pawn gain also misses the meaningful-progress floor. This small
signal is retained; the criterion was not weakened after observing it.

No candidate passed. As preregistered, there was no fresh test, no additional
training seed, no extended training and no larger corpus. The planned conditional
fresh-game ensemble evaluation remains UNEXECUTED. The next route is a bounded
material-error mechanism audit using existing development evidence before choosing
another model or source domain; see `../S005-handoff.md`.

## Reproducibility and validation

The plan, fold assignments and split hashes were committed before training at
`82a653ad0e0f68829934ec4c42e7faa31d812379`. Six fits ran from 13:50:04 to 13:50:27
UTC, about 23 seconds. Full inner-validation histories, selected checkpoint hashes,
numerical source hashes and execution commit are in `S004-training.json`.
`S004-results.json` contains every scale, fold, seed and gate score.

The independent audit reloaded all six checkpoints, verified their hashes,
matched inner-validation metrics, reproduced all saved outer predictions exactly,
checked every partition's source and canonical separation, verified every source
game occurs as outer once, recomputed candidate mean-game differences, and
independently obtained the failed gate. Audit status PASS means the records and
calculation passed verification; it does not mean the scientific gate passed.

All 31 tests pass, including fold-separation checks and synthetic cases that
verify the meaningful-gain floor, rejection when one fold worsens, and equal
weight per game. A completed-run resume verifies hashes and preserves the frozen
training and result files without retraining. Every inventoried S002/S003 artifact
remains hash-identical. Their hosted CI checks were green at S004 startup.

## Durable local artifacts and closeout

- Fold datasets: `C:\chess-intelligence-lab\data\splits\s004-cv-v001`.
- Six checkpoints: `C:\chess-intelligence-lab\checkpoints\s004\fold-{0,1,2}-seed-{7,19}.pt`.
- Plan, training, OOF predictions, result, audit and logs:
  `C:\chess-intelligence-lab\runs\s004`.
- Small shareable copies and hashes: `docs\experiments\S004-*.json`.

The checkpoints are development-fold models, not a promoted deployment model.
Their saved weights support inference; optimizer state for exact mid-epoch resume
is not included. Runtime data, checkpoint binaries and logs remain Git-ignored.
The owner's C:-only storage preference persists; no independent backup was made.
No other repository was modified. No training or teacher process is left running.
