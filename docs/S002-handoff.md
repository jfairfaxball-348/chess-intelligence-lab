# S002 — LOCAL Windows session: persistent Stockfish training begins

**Execution mode: LOCAL, on the owner's Windows laptop. Do not execute S002
training in a disposable cloud coding environment or GitHub Actions runner.**
The owner is switching from cloud S001 development to local compute for S002.
This is a practical storage/compute decision, not a change to the scientific
aims. Cloud GitHub Actions may continue to run short CI tests.

**Repository:** https://github.com/jfairfaxball-348/chess-intelligence-lab
**Authoritative code:** latest committed `main` at the start of this session.
**S001:** GitHub Actions passed 14 tests and the synthetic training/UCI smoke.
It did *not* demonstrate useful model learning: the 32/12-position smoke test's
validation MAE rose from 7.212671 to 7.238849 pawns. S002 is the first
genuine Stockfish-supervised experiment, not a repeat of S001 setup.

## Start-up instructions for the locally executing coding agent

1. Confirm this session actually has local command/terminal access to the
   owner's Windows laptop, with a persistent working directory and disks.
   Merely stating "local" in this document does not make a cloud session local.
   If the agent is still cloud-hosted, do **not** start costly dataset creation:
   report that the owner must open this repository in a local Codex/Claude Code
   session and run the S002 handoff there.
2. Use the owner's existing local clone if present (GitHub Desktop is fine).
   Fetch/pull latest `main` carefully, inspecting for uncommitted local changes;
   don't overwrite local work, reset branches, or discard local data. If no
   clone exists, clone this repository to a durable laptop folder.
3. Read `README.md`, `pyproject.toml`, `src/chesslab/data.py`,
   `src/chesslab/teacher.py`, `src/chesslab/train.py`,
   `src/chesslab/uci.py`, and this handoff before editing anything.
4. Record actual hardware. Do not assume a discrete GPU or particular VRAM,
   available disk or RAM. PowerShell example:

       Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
       Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
       Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM
       Get-PSDrive -PSProvider FileSystem | Select-Object Name,Free,Used
       nvidia-smi

   `nvidia-smi` may be unavailable. Also record Python, PyTorch, Windows,
   Stockfish and CPU/GPU capabilities before sizing any experiment.
5. Create a project-local virtual environment on the laptop. PowerShell commands
   from repository root (use `py` if the Python launcher is installed):

       py -3 -m venv .venv
       .\.venv\Scripts\python.exe -m pip install -e ".[train,dev]"
       .\.venv\Scripts\python.exe -m pytest -q

   Use the official PyTorch wheel appropriate to the detected CPU/GPU,
   rather than assuming CUDA is present. Fix actual dependency errors if any.

## Persistent local storage: non-negotiable

All training data and trained weights must survive agent-session completion,
terminal closure, reboot, fresh Git checkout, and subsequent S003+ sessions.

Use stable paths in the *same existing local checkout*, unless volume size
requires an explicitly configured external data root. Proposed layout:

    data/raw/               # source PGNs, FENs and provenance
    data/splits/            # train / validation / held-out test, grouped by game
    data/teacher/           # genuine Stockfish-annotated data
    checkpoints/s002/       # model state files, never overwritten silently
    runs/s002/              # config, metrics, logs, PGNs, timing results
    docs/experiments/       # small shareable experiment reports and manifests

The first five paths are Git-ignored; they are **not** backed up by GitHub.
The final `docs/experiments/` location is for small, non-sensitive textual
manifests/reports, not raw training data. Do not assume ignoring files makes
them persistent if the agent uses a temporary sandbox. Check paths and
free disk space and write an actual file; verify it survives re-opening.

- Version datasets (e.g. `teacher-v001`), never regenerate/relabel into the
  same files silently. Version model checkpoints (`model-v001.pt`, etc.).
- Record SHA-256 hashes, sample counts, generation seeds, source/licence,
  source game IDs, split policy, teacher executable hash/version, depth/nodes
  and options, architecture version, hyperparameters, training start/stop,
  validation and test results, Git commit SHA and absolute local paths.
- Put small manifests, commands and comparable metrics in Git.
  Do **not** commit PGNs, large JSONL corpora, checkpoints or logs.
- Set up or explain a **second durable backup location** (separate volume or
  chosen external storage) for valuable datasets/checkpoints. A local disk
  without backup can still fail; verify the backup by hash. Never send large
  data to a third party without owner agreement.
- At the end, explicitly state where data and weights live so a later
  session can resume without reannotation or retraining.

## S002 scope: first *real* teacher-supervised experiment

1. Verify the S001 local smoke and tests. Then install a native Stockfish
   executable appropriate to this machine, record exact version, location and
   SHA-256; keep teacher invocation entirely separate from model inference.
2. Produce a small but useful initial corpus of **genuine Stockfish-labelled**
   legal positions. Get PGNs from licensable/attributable sources or generate
   reproducible engine games locally; record provenance. Sample diverse
   openings, middlegames, endgames and evaluations. Do not claim random
   playout positions are representative of normal chess.
3. Split by **source game before extracting positions**, preserving a final
   unseen test set. Deduplicate across partitions (including recognised
   symmetric/transposition duplicates); prevent any training/test game leakage.
4. Inspect and, if needed, improve the existing `annotate` and dataset
   tools to support stable incremental/batched output, failure recovery,
   complete manifests and reproducible annotations. Start with a short
   50–100-position benchmark; then size a first trainable corpus based on
   measured local time/storage (aim for thousands, not millions, initially).
   The current adapter accepts FEN/JSONL as input and records teacher PV,
   centipawns, mate scores separately, depth/nodes and engine metadata.
5. Train the existing PyTorch scalar evaluator on **real Stockfish labels**,
   save the exact dataset split and a versioned checkpoint. Evaluate on the
   held-out test set only after model selection based on validation results.
   Compare against the untrained network and material-only evaluator.
   Neural predictions must never call Stockfish.
6. If practical, run an initial *legal* equal-budget material-versus-neural
   UCI match and save PGNs, colours, openings, timings and W/D/L. The S001 UCI
   engine is currently synchronous, fixed-depth, and not clock-aware:
   improve its UCI/timing support before claiming a valid timed-match Elo.
   Otherwise explicitly mark match strength testing as a remaining gate.
7. Diagnose failed learning honestly: the tiny S001 network may be unsuitable
   and the real corpus may expose flaws. Report predictive accuracy and
   playing strength separately; do not overstate either.
8. Commit/push code, tests and small textual manifests through the normal Git
   workflow, without touching the mathematical proof or Playground repos.

## Success criteria and stop condition

S002 should finish with:
- Working **local** Python/PyTorch/Stockfish environment and tests.
- Persistent, game-separated Stockfish-labelled train/validation/test files,
  with provenance, counts, hashes, reproducible teacher configuration and
  incremental recovery rather than an ephemeral GitHub Actions-only artifact.
- At least one trained, reloadable, versioned neural checkpoint stored locally.
- Honest measured held-out metrics versus material and untrained baselines.
- If feasible, a small correctly controlled chess-playing evaluation; if not,
  exact blockers and next steps.
- Short committed report plus a precise S003 handoff pointing to durable local
  datasets, checkpoints and resume commands. Confirm a backup policy.

Start with a bounded, measurable pilot. Do not launch a huge unattended training
run until storage, recovery and measured throughput are understood. Avoid
bureaucratic session protocols; prioritise experimental work and persistence.

## Scientific and project boundaries

S001 results establish *technical operability*, not chess-playing strength.
Formal endgame research (`engame-chess`) is proof-first and independent;
KRK is not globally solved. The Maths Endgame mode in
`engame-chess-playground` must never consume experimental neural moves as
mathematically proved results. Do not modify either separate repository.

Long-term options include stronger value/policy models, independent search,
reinforcement learning, and manually authorised theorem-scoped training
signals. Do not prematurely implement them in S002.

**Begin S002 in the laptop's local coding-agent environment, not this cloud chat.**
