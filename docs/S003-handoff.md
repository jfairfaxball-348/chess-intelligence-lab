# S003 - local successor handoff (FUTURE / UNEXECUTED)

S002 completed its bounded local experiment. Read `experiments/S002-report.md`
and its adjacent JSON manifests before new work. The neural checkpoint learns
some signal but loses to the material baseline on prediction error. Strength
is unknown. No further training or search is running after S002 closeout.

## Reuse existing local artifacts first

Working directory: `C:\chess-intelligence-lab`.
Python: `C:\chess-intelligence-lab\.venv\Scripts\python.exe`.
Do not recreate or clean ignored directories. Check the committed artifact
inventory and actual SHA-256 before reuse. Raw games, labelled splits and both
checkpoints are already present; neither annotation nor training is required
to load the selected engine.

PowerShell from this checkout:

    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe scripts\audit_s002.py
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    .\.venv\Scripts\python.exe -m chesslab.uci --checkpoint checkpoints/s002/model-v001.pt

Example UCI input:

    uci
    isready
    position startpos moves e2e4 e7e5
    go depth 1
    quit

Use `model-v001.pt` (selected by validation); v002 is retained for provenance.
The checkpoint contains model/encoding versions and weights, not optimizer state;
this supports inference and reproducible restart, not exact mid-epoch continuation.
Full training histories are in `runs/s002/result.json`; Git has a small copy in
`docs/experiments/S002-results.json`.

To reproduce from an empty, separately versioned data/checkpoint location, inspect
`scripts/s002_local.py` and configure new versioned paths first. The S002 commands
were, in order:

    .\.venv\Scripts\python.exe scripts\s002_local.py generate --games 80
    .\.venv\Scripts\python.exe scripts\s002_local.py benchmark
    .\.venv\Scripts\python.exe scripts\s002_local.py label
    .\.venv\Scripts\python.exe scripts\s002_local.py train

Do not rerun `train` against the existing checkpoint paths: it intentionally
refuses overwrite. A failed individual annotation can resume with exactly the
same input, engine and settings; for example:

    .\.venv\Scripts\python.exe -m chesslab.cli annotate --input data/splits/engine-v001/train.jsonl --output data/teacher/teacher-v001/train.jsonl --engine artifacts/stockfish/sf19/stockfish/stockfish-windows-x86-64-universal.exe --depth 8 --threads 1 --hash-mb 32

`*-provenance.jsonl` files contain source game/ply metadata and are the actual
training/evaluation files. The raw annotation outputs and their sidecar resume
manifests must remain together. Keep versions immutable; use teacher-v002 and
new checkpoint filenames for changed settings. New Python environments can use
the dependency versions in `experiments/S002-requirements.txt`; install the local
project with `pip install -e .` rather than treating the recorded editable base
commit line as the completed S002 source revision.

## Suggested bounded successor

1. Diagnose the material-versus-neural gap on training/validation only. Compare
   side-to-move normalization, a material residual target and loss choices as
   hypotheses; do not assume any is an established fix. Keep new architectures
   or long training separately scoped. Preserve S002's measured negative result.
2. The S002 test set is now observed. Do not use it for later model selection;
   reserve new source-game-separated data as a fresh final test for new tuning.
3. If strength testing is chosen, first implement asynchronous, clock-aware UCI
   search with reliable `stop`, deadlines and legal fallback. Validate budget
   enforcement and then run paired equal-budget material/neural games with PGNs,
   colors, openings, timings, W/D/L and uncertainty. Fixed depth is not equal time.
4. Only C: was available. Owner chose local-only storage. There is no independent
   backup. If another volume becomes available, copy raw data, split/teacher files,
   checkpoints, Stockfish licence/version and run metadata; hash-verify the copy.

Do not modify `engame-chess` or `engame-chess-playground`. These experimental
predictions establish no mathematical endgame proof or complete WDL classification.
