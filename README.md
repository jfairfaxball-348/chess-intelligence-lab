# Chess Intelligence Lab

Independent chess-AI research using teacher-guided learning, independent search,
eventual reinforcement learning and separately authorised mathematical signals.

**This is an early experimental engine, not a competitive chess program.**
The synthetic S001 teacher is just a material heuristic, NOT Stockfish.
Neural inference, search and UCI never call Stockfish or a tablebase.

## Boundaries

- [Mathematical endgame research](https://github.com/jfairfaxball-348/engame-chess):
  separate proof-first repository. K/K, K/B/K and K/N/K drawing work is frozen;
  KRK is **not** completely proved. No automatic theorem import.
- [Chess Playground](https://github.com/jfairfaxball-348/engame-chess-playground):
  independent visual frontend. Only approved completed mathematical results
  appear in its Maths Endgame mode. Experimental UCI engines belong solely in
  its separate Engine Playground mode.
- This repository trains, evaluates and packages independent engine prototypes.
  No frontend, Vercel, or other repository is modified here.

## Setup

Use Python 3.10+ in a terminal (Windows PowerShell or a cloud Linux host).

    python -m venv .venv
    # Windows PowerShell: .\.venv\Scripts\Activate.ps1
    # Linux/macOS: source .venv/bin/activate
    python -m pip install -e ".[train,dev]"
    python -m pytest -q

If installing PyTorch needs a platform-specific wheel, use the official
PyTorch install selector, then install the base/dev extras separately.
Core material/search/UCI tests need only python-chess and pytest.

## First experiment: deliberately synthetic

    python -m chesslab.cli smoke --out runs/s001 --train-count 96 --val-count 24 --epochs 3 --seed 7
    python -m chesslab.cli evaluate --checkpoint runs/s001/tinyconv.pt --dataset runs/s001/val.jsonl

This writes separate train/validation JSONL datasets, a PyTorch checkpoint and
a JSON report. All positions are legal random-playout samples and labels are
material-centipawn evaluations from the side-to-move perspective. Validation
and training have no identical full FENs. Model outputs are in pawn units.
Lower label error is NOT evidence of higher playing strength.

The 19-plane board encoding specifies absolute white/black pieces, turn,
castling rights, en-passant square and a clipped halfmove-clock plane.
The model is a deliberately tiny CNN, designed to be replaceable.

## Basic UCI engine

Material search, no PyTorch dependency:

    python -m chesslab.uci

Neural search using a checkpoint (independent of Stockfish at inference):

    python -m chesslab.uci --checkpoint runs/s001/tinyconv.pt

Example UCI input:

    uci
    isready
    position startpos moves e2e4 e7e5
    go depth 2
    quit

The current engine supports the UCI handshake, positions, legal-move search
and fixed depth. It is *synchronous and not clock-aware*: do not use for
rated timed games until time management, stop/ponder and asynchronous search
are implemented. Its depth-limited negamax lacks quiescence and transpositions.

## Optional genuine Stockfish labels

Stockfish is a separate optional native executable. Example:

    python -m chesslab.cli annotate --input data/pilot.fens --output data/pilot-stockfish.jsonl --engine "/path/to/stockfish" --depth 8 --threads 1 --hash-mb 32

Input: one FEN per line or JSONL objects with a fen key. Output metadata
includes the teacher identity, configured depth, reported search depth/nodes,
principal variation and best move. Mate scores have a separate mate_in value
and NULL centipawn target; no pseudo-centipawn mate scores are fabricated.
A single training run rejects mixed synthetic and real teacher sources.

## Research roadmap

- **S001:** Synthetic data split, transparent material baseline, trained tiny
  CNN, optional Stockfish adapter, standalone fixed-depth UCI, CI tests.
- **S002:** Inventory local hardware; capture provenance-controlled real
  Stockfish data; train/test by source game; compare at matched compute and
  legal paired games with timings and uncertainty.
- **Later:** Policy/value networks and search comparisons; self-play and
  checkpoint tournaments; manually authorised, theorem-scoped mathematical
  signals with explicit proof-versus-prediction separation.

See [S001 experiment](docs/S001-experiment.md) and
[S002 handoff](docs/S002-handoff.md). Datasets, checkpoints and experiment
outputs remain local or CI artifacts; Git tracks only source and manifests.
