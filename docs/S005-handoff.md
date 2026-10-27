# S005 - material-error mechanism audit (FUTURE / UNEXECUTED)

S004 is closed with its preregistered gate FAILED. Read `experiments/S004-report.md`,
`S004-plan.json`, `S004-results.json` and `S004-audit.json` first. This handoff
selects the preregistered pivot: inspect why material disagrees with the teacher
before proposing more CNN training or a larger corpus.

## Current boundary

Three outer game folds, two seeds and fixed residual shrinkage gave material
mean-game MAE 2.116065 versus selected half-correction 2.100821. The 0.015244-pawn
gain is below the preregistered 0.05 threshold, one fold worsened, and the interval
[-0.032033, +0.001182] includes zero. A quarter-correction showed a smaller
consistent development signal but also missed the threshold. Preserve both
findings; do not recast S004 as a confirmed material-beating model.

No S004 final-test games or labels were generated. The conditional final evaluation
is UNEXECUTED. Both prior final tests are already observed. Playing strength is
unknown and the UCI engine still lacks timed/stop support. Do not infer Elo or
complete endgame WDL results.

## Verify local state first

Use the same `C:\chess-intelligence-lab` checkout and `.venv\Scripts\python.exe`.
Inspect local changes, remote branches and predecessor PR states before writes.
S004 is stacked on S003, which is stacked on S002; reconcile the live GitHub
state rather than assuming these have merged. Do not reset or clean ignored data.

    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe scripts\audit_s004.py

Existing development source files:

    data/teacher/teacher-v001/train-provenance.jsonl
    data/teacher/teacher-v001/val-provenance.jsonl

S004 fold files are in `data/splits/s004-cv-v001`. Six fold checkpoints and complete
records live in `checkpoints/s004` and `runs/s004`. Artifact inventories provide
absolute paths, sizes and SHA-256 hashes. Historical S002/S003 files remain intact.

The original S004 command sequence was `scripts/s004_local.py prepare`, `train`,
`summarize` using the local venv Python. A completed rerun verifies existing
records; it must not overwrite checkpoints or silently change the frozen plan.
The `train` command skips checkpoint/result pairs after hash verification.

## Suggested bounded S005 scope

Do not start another neural fit by default. Using only the existing original
72-game development corpus:

1. Quantify signed and absolute material-versus-teacher residuals by game,
   material/phase, side to move, check status and immediate capture availability.
   Report denominators and game-balanced comparisons; keep mate labels separate.
2. Replay a bounded, deterministic set of high-error and matched low-error
   positions from their source PGNs. Distinguish material-counting/POV/provenance
   bugs, shallow-teacher instability, and tactical or positional residuals.
   Legal position/replay checks must precede a model conclusion.
3. If deeper teacher checks are useful, preregister the selection, cap and decision
   criterion before annotation. At most 128 existing development positions at
   depth 12 initially, using the same hashed native Stockfish executable and
   new versioned output paths. Include matched controls; preserve depth-8 labels
   and never silently relabel them. Do not inspect old final-test errors to choose
   these positions.
4. Produce a concrete evidence-based next target, or an honest unresolved result.
   Possible alternatives include a tactical/search feature baseline, a changed
   teacher/source domain or conservative residual correction. None is an
   established fix. State a measurable criterion and bounded pilot before any
   future training. Do not use throughput or a tiny score decrease as progress.

Save the audit, selected FEN/source-game provenance and teacher comparisons under
new `runs/s005` and `data/.../s005-*` paths, with small committed reports and
hashes. S006 remains future until S005 is closed and a next design is authorized.

The local-only/no-independent-backup preference persists. Do not upload large data
or weights, or touch `engame-chess` / `engame-chess-playground`. Their proof status
and theorem-import boundaries are separate from these experimental predictions.
