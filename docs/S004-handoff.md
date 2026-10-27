# S004 - local successor handoff (FUTURE / UNEXECUTED)

Read `experiments/S003-report.md`, `S003-plan.json`, `S003-results.json` and
`S003-audit.json` before new experiments. S003 is closed. No training or annotation
is left running. Work remains local in `C:\chess-intelligence-lab`.

## Current result and authority

S003 trial 7 (normalized material residual, MSE) was chosen by validation:
2.202618 pawns MAE versus material 2.235201. Fresh-game MAE reversed the advantage:
S003 2.139673, material 1.882747, S002 3.458498, over 750 finite labels from
20 games. The paired whole-game bootstrap interval for excess neural error is
[0.128318, 0.408669] pawns. Retain this negative result. Playing strength remains
unknown. Neither S002's old test nor S003's fresh test is now unseen.

S003 builds on S002 PR #2. If either PR remains open, use their branch tips and
inspect GitHub state before integrating. Do not reset or discard local artifacts.
The completed S003 numerical experiment ran at commit
`4f17d522530fa40e13ba52e413f2799af5c2384a`; reports and audits were committed afterward.

## Verify and reuse, without retraining

PowerShell from the persistent checkout:

    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe scripts\audit_s003.py
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    .\.venv\Scripts\python.exe -m chesslab.uci --checkpoint checkpoints/s003/model-v007.pt

Then send `uci`, `isready`, `position startpos`, `go depth 1`, `quit` as separate
input lines. This is a legal fixed-depth smoke only. Use material as the current
predictive baseline; do not promote S003 as a material-beating engine.

S002 files remain under `data/raw/engine-v001`, `data/splits/engine-v001`,
`data/teacher/teacher-v001`, `checkpoints/s002` and `runs/s002`.
S003 fresh data lives under the matching `data/{raw,splits,teacher}/s003-holdout-v001`
directories. S003 checkpoints and logs are in `checkpoints/s003` and `runs/s003`.
Artifact inventories give absolute paths, byte sizes and hashes. Keep raw teacher
JSONL plus resume sidecars together; use provenance-enriched JSONL for training.

The original S003 commands were `scripts/s003_local.py prepare`, `train`, then
`final`, using the local venv Python. Do not rerun `final`: it rejects an existing
result and the final test has already been observed. `train` verifies and skips
completed trial checkpoints using `runs/s003/training.json`; the saved selected
checkpoints are inference states, not exact mid-epoch optimizer-resume states.
Do not overwrite S002/S003 paths for new experiments.

For a new bounded training command, the CLI now supports:

    .\.venv\Scripts\python.exe -m chesslab.cli train --train data/teacher/teacher-v001/train-provenance.jsonl --val data/teacher/teacher-v001/val-provenance.jsonl --checkpoint checkpoints/s004/model-v001.pt --epochs 20 --seed 7 --normalize-stm --material-residual --loss mse --include-initial

This example is a syntax reference, not an instruction to start S004 training
without a new experiment design. Checkpoint versions store encoding, residual
configuration and weights so UCI uses the same representation as training.

## Suggested next bounded decision

Before enlarging the dataset or training longer, address selection robustness:
use group-by-game development folds and fixed budgets to test whether any residual
correction beats material consistently, rather than winning narrowly on one
validation split. Multiple training seeds or conservative residual shrinkage are
hypotheses, not established fixes. Preregister the choices and success criterion
before running; include the exact material baseline and initialization fallback.
If the criterion fails, close the experiment and pivot instead of increasing
throughput. Reserve a separately seeded, game-separated final test only after the
development design is fixed; do not tune after inspecting that final result.

If the owner instead selects playing-strength work, implement and test asynchronous
clock/stop/deadline support first, then run legal paired equal-budget games with
PGNs, openings, colors, timings and W/D/L. Fixed depth is not equal time.

The owner's local-only storage preference persists. No second volume/backup is
available. Do not upload data or checkpoints to a third party. Do not modify
`engame-chess` or `engame-chess-playground`; no experimental prediction is a
mathematical proof, a complete WDL classification or an approved theorem import.
