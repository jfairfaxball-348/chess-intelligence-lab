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

## Binding research direction (owner steering, 27 October 2026)

**North star:** Develop a **playable, progressively improving, original chess AI**. Train from strong external teachers (initially Stockfish) to acquire general chess knowledge; run **our own inference and search without consulting Stockfish**; establish genuine playing strength; improve via self-play/reinforcement learning; then investigate whether **formally verified mathematical chess knowledge** improves generalisation, sample efficiency or search. Eventually export local UCI models for the **separate Engine Playground**. An Elo or W/D/L improvement at a fair total compute budget matters more than a few hundredths of a pawn in teacher-score error.

**Why S005 exists:** The tiny ~15.8k-parameter CNN and 80 shallow teacher-guided games were *scaffolding*, not a commitment to indefinite scalar centipawn regression. S002 demonstrated learning but remained worse than material on unseen positions; S003 improved its predecessor but remained worse than material on fresh games; S004's small out-of-fold correction **failed** its preregistered meaningful-progress gate. Preserve all negative results. S005 is a **bounded concluding diagnosis** of this pilot, not licence to run S006/S007 as further micro-tuning of the same CNN and MAE. Identify concrete data/label/representation errors or credible alternatives; then **pivot to the broader programme**.

### Future work, in scientific priority order

1. **Playable evaluation first.** Make the stand-alone UCI engine correct under real time controls: asynchronous search or reliably interruptible `stop`, clocks/movetime, terminal adjudication, legality, fair timing, and crash-safe match logging. Benchmark material versus saved neural checkpoints in paired, colour-swapped openings with equal *total* wall-clock and measured CPU/GPU/search resources. Report W/D/L, time, nodes and uncertainty. Small runs are smoke tests, **not Elo evidence**. Keep Stockfish an external opponent/teacher, never an inference dependency.
2. **Actual foundation training, not just centipawn fitting.** Design larger and genuinely diverse, source-game-separated teacher data (openings, mistakes, tactics, human/engine games, defensively difficult positions, endings) with pinned Stockfish version, PV/move candidates, depth/nodes, calibrated WDL *when available*, draw/mate treatment and provenance. Compare scalar values with **policy + value/WDL** models and search-compatible efficient networks; retain simple controls. Use new unseen holdouts, not S002/S003 observed tests. Treat shallow depth-8 CP targets as noisy, not absolute truth.
3. **Independent learning loop.** Train a baseline checkpoint that runs as a UCI player, then build self-play generation, historical-opponent/checkpoint competitions, game-outcome reward, search-improved targets and reproducible reinforcement-learning updates. Demonstrate improvement over a frozen supervised checkpoint under matched resources, not just improved imitation loss. Do not attempt AlphaZero-scale self-play on an unmeasured budget.
4. **Mathematical learning hypothesis.** Maintain a *future*, explicit adapter for manually approved immutable theorem artifacts from `engame-chess`. Test theorem-derived properties (e.g. certified rook safety/confinement/progress within exact hypotheses) as auxiliary labels/features and use rigorous with/without-theorem ablations. **KRK is not yet globally solved**; never treat partial results as full outcome/DTZ labels. No direct import into the Playground's Maths Endgame, no tablebases in proof generation, and no edits to the theorem repo. The theorem programme and neural programme stay independent.
5. **Model and search flexibility.** Keep replaceable position encoders, scalar evaluators, policy/value heads, search, teacher adapters, theorem-labelled features, game/tournament harness and UCI packaging. Investigate compact alpha-beta-friendly NNUE-like nets versus batched policy/value/MCTS or transformers based on measured training/inference costs—not on architectural fashion. Do not build a large orchestrator without evidence of incremental benefit.

**Priority test for every new session:** What becomes better at *playing chess* or at learning generalisable strategy, how will we measure it fairly, and why is this the least-complex experiment that answers the question? Pure label-loss experiments remain legitimate for diagnosis and pretraining, but must not become the research objective. Negative experiments should inform a new hypothesis, not lock us into repeated small retunings.

**Known S002 hardware:** Windows 11 laptop, i7-11800H (8 cores/16 threads), ~16 GB RAM, RTX 3050 Ti Laptop (4 GB VRAM); current PyTorch environment was CPU-only. Measure CPU/GPU throughput before moving training to CUDA, upgrading architectures or ordering new hardware. Keep persistent ignored datasets/checkpoints intact; no independent backup is currently documented.

### Required S005 closeout and S006 handoff

- Complete only the bounded audit below, preserving fixed provenance, honest uncertainty and frozen results. Do not prolong S005 with unfocused extra runs.
- Give an evidence-based explanation for the failed scalar-distillation pilot, or record that it remains unresolved. Separate bugs, shallow teacher noise, data coverage, representation limits and genuine tactical/positional residuals.
- **Install an S006 prompt** whose primary goal is an interruptible, clock-aware local UCI engine plus a reproducible paired match harness, unless S005 uncovers a critical correctness blocker that must be fixed first. Include measurable smoke tests, calibrated strength-testing criteria, full compute accounting and failure handling. Do not make S006 another residual-scaling sweep.
- Explicitly preserve the longer roadmap: broad teacher-labelled **policy/value** foundation training → independent self-play RL → approved theorem-guided learning experiments. Keep these goals visible in the README/next handoff without building them prematurely.
- The **Playground is a separate display and playing interface**; local model integration occurs through a future UCI/localhost bridge. The Maths Endgame's strict theorem gates remain unchanged. Never alter either other repository from this lab session.

## Verify local state first

Use the same `C:\chess-intelligence-lab` checkout and `.venv\Scripts\python.exe`.
Inspect local changes, remote branches and predecessor PR states before writes.
S002, S003 and S004 have been incorporated into main by a non-forcing fast-forward
when this prompt was installed. **Verify live GitHub and local branch state** rather
than assuming a local checkout is current. Fetch latest main safely; preserve
any uncommitted S005 work and do not reset or clean ignored data.

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
hashes. S006 remains future until S005 is closed. The owner has authorised the research
steering above; install its engineering-and-playing-strength prompt at closeout.

The local-only/no-independent-backup preference persists. Do not upload large data
or weights, or touch `engame-chess` / `engame-chess-playground`. Their proof status
and theorem-import boundaries are separate from these experimental predictions.
