# S003 - bounded local diagnosis and fresh holdout

S003 completed locally on 27 October 2026. Side-to-move normalization and a
material residual substantially improved on the S002 neural model, but the
validation-selected S003 model still lost to material on fresh games. This is
predictive evaluation only. No playing-strength gain or Elo is established.

## Fresh held-out result

| Evaluator | MAE, pawns | Neural RMSE, pawns |
| --- | ---: | ---: |
| Material only | 1.882747 | - |
| Selected S003, normalized material residual / MSE | 2.139673 | 2.855170 |
| Frozen S002 checkpoint, same fresh games | 3.458498 | 4.765829 |

The test contains 777 positions from 20 new source games: 750 finite CP labels
and 27 explicit mate labels, which remain excluded from scalar metrics. Relative
to material, S003 adds 0.256926 pawns of MAE. A paired bootstrap resampling whole
games (5,000 draws, seed 303) gives a 95% percentile interval of [0.128318,
0.408669] pawns for neural-minus-material error. Positive values mean worse
neural prediction. This describes sampling uncertainty within this shallow-engine
game source, not uncertainty about human chess or playing strength.

No hyperparameters were changed after this result. Both old and fresh test sets
are now observed and must not be used as future final unseen tests.

## Diagnosis and fixed experiment

S002 artifacts were verified against their committed hashes; all 17 predecessor
tests passed, and predecessor PR #2 had green hosted CI. Existing training and
validation data were reused without reannotation or overwrite.

S002 training MAE was 2.057825 versus material 2.101459, but validation MAE was
2.822334 versus material 2.235201. This establishes a train/validation gap;
it does not isolate the cause. Validation material error was similar for white
to move (2.175816) and black to move (2.288535), so a simple global turn-sign
mistake was not established.

The plan and reserved-game manifest were committed before the completed training
run. Eight trials form the full 2 x 2 x 2 comparison of absolute/normalized input,
direct/material-residual prediction, and MSE/Huber(delta=1 pawn) loss. The CNN
architecture and 15,841 parameters were retained: width 32, AdamW learning rate
0.002, batch 32, seed 7, one CPU thread, 20 epochs per trial. This is one training
seed and one development split, not a multi-seed robustness study.

Normalization uses color swap plus rank reflection for black-to-move boards so
the mover is always encoded as white. It preserves castling, en-passant and
clock state without changing the source board. Residual prediction is
`material_stm(board) / 100 + neural_correction(board)` in pawn units. The final
layer starts at zero for residual models, giving the exact material baseline at
epoch zero. Huber is applied to the prediction error with delta 1 pawn.

All trials select minimum validation MAE, including initialization at epoch zero;
trial selection then uses minimum validation MAE, with stable trial order for
ties. This deliberately lets a harmful correction fall back to material. It also
means a saved checkpoint can be an untrained initial state.

| Trial | Input | Prediction | Loss | Selected epoch | Validation MAE |
| --- | --- | --- | --- | ---: | ---: |
| 1 | Absolute | Direct | MSE | 20 | 2.822334 |
| 2 | Absolute | Direct | Huber | 19 | 3.009174 |
| 3 | Absolute | Material residual | MSE | 0 | 2.235201 |
| 4 | Absolute | Material residual | Huber | 0 | 2.235201 |
| 5 | Normalized | Direct | MSE | 16 | 2.388872 |
| 6 | Normalized | Direct | Huber | 19 | 2.340533 |
| 7 | Normalized | Material residual | MSE | 17 | 2.202618 |
| 8 | Normalized | Material residual | Huber | 9 | 2.218412 |

Trial 7 was selected before fresh test labelling. Its small validation advantage
(0.032583 pawns) reversed on fresh games. Normalization helped the direct model
on validation; an absolute-input residual selected no correction. Huber had no
consistent benefit. These matched comparisons support a useful implementation
diagnosis, but do not establish a general winning model or justify scaling up.

Completed training took about 46 seconds from 13:40:27 to 13:41:13 UTC. Fresh
labelling and final evaluation finished at 13:41:37 UTC. The exact plan, per-epoch
histories, selected checkpoint hashes, execution commit and numerical source
hashes are in `S003-results.json` and adjacent manifests. The completed training
code commit is `4f17d522530fa40e13ba52e413f2799af5c2384a`.

## New source games and audit

Twenty local Stockfish-guided games were assigned entirely to the new test set
before extraction. Generation seed 30361027 plus game index; depth 4, MultiPV 3,
move-choice weights 8:2:1, one thread, 32 MiB hash, 240-ply cap. The teacher is the
same hashed Stockfish 19 executable as S002, depth 8, one thread, 32 MiB hash,
cleared per position. Teacher labels were not read until selection was frozen.
The corpus is engine-generated and not representative of human chess.

Up to 40 positions per game were sampled evenly over legal nonterminal positions
from ply eight. Canonical transpositions and recognized color/rank/file symmetries
were excluded against every S002 split and within the new test. S002's old test
positions were used for duplicate exclusion in the new experiment; its metrics
were not used for model selection.
Source game IDs are in a separate namespace. The audit checked no duplicated full
game trajectories across all 100 games, zero canonical overlap, source FEN/ply
identity, legal source moves and teacher PVs, teacher depths and hashes, and
all eight checkpoint hashes. It passed.

The first completed trial exposed a report serialization bug introduced in S003:
the configured loss name had been shadowed by the batch-loss tensor. The fix has
a regression check. The failed report log and initial checkpoint were retained;
the repeated trial's weights were independently verified identical. This was a
reporting failure, not a new model-selection trial or a discarded negative result.

All 27 tests pass. Six UCI checks loaded both the S002 and selected S003
checkpoints and returned legal moves, including black-to-move positions.
Versioned checkpoint metadata distinguishes normalized encoding and residual
models; old S002 checkpoints still load. Inference uses the material formula and
CNN only, without Stockfish. Timed UCI strength testing remains unexecuted because
search is still synchronous and lacks deadline/stop support.

## Durable local artifacts and closeout

- Raw games: `C:\chess-intelligence-lab\data\raw\s003-holdout-v001`.
- Reserved FENs and manifest: `C:\chess-intelligence-lab\data\splits\s003-holdout-v001`.
- Teacher labels: `C:\chess-intelligence-lab\data\teacher\s003-holdout-v001`.
- Eight trial checkpoints: `C:\chess-intelligence-lab\checkpoints\s003\model-v001.pt`
  through `model-v008.pt`. Selected: `model-v007.pt`.
- Failed-report checkpoint: `checkpoints\s003\model-v001-unrecorded.pt`.
- Full plan, results, audit, logs and recovery evidence: `C:\chess-intelligence-lab\runs\s003`.

All original S002 data and checkpoints remain hash-identical. New large/local
artifacts remain ignored. Git contains only code, tests and small textual records.
The owner chose C:-only storage in S002; no independent backup has been made.
No other repository was changed. S004 is future work; see `../S004-handoff.md`.
