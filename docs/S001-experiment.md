# S001 first experiment: can a tiny network imitate material evaluation?

**Research question.** Can a model trained only on generated, legal chess positions
reduce held-out error against an easily reproducible material-value teacher?

**Important limitation.** The teacher is a simple handcrafted material heuristic,
NOT Stockfish. The experiment measures pipeline correctness and learnability, not
playing strength, general chess understanding, or superiority to the teacher.

## Protocol

1. Sample 96 training and 24 validation legal nonterminal positions by seeded
   random playouts, excluding exact FEN duplicates across splits.
2. Evaluate each position with the deterministic material baseline from the
   perspective of the player to move.
3. Use 19-plane absolute-coordinate encoding; a small 2-convolution-layer
   network predicts a scalar pawn value. Train with MSE/AdamW for three epochs.
4. Record model architecture/version, seed, parameter count, checkpoint SHA-256,
   label source, held-out MAE/RMSE and material baseline MAE.
5. Test UCI search separately with the same legal-move interface.
6. Compare equal-depth search legally; do not claim strength improvements
   from a validation-loss reduction.

**Commands.**

    python -m chesslab.cli smoke --out runs/s001 --train-count 96 --val-count 24 --epochs 3 --seed 7
    python -m chesslab.cli evaluate --checkpoint runs/s001/tinyconv.pt --dataset runs/s001/val.jsonl
    python -m chesslab.uci --checkpoint runs/s001/tinyconv.pt

**Success gates.** CI tests pass; zero cross-split exact-FEN overlap; training
produces finite metrics; a checkpoint reloads; independent inference returns
a centipawn evaluation; UCI returns legal best moves; no Stockfish executable
is needed anywhere in this synthetic experiment.

**Outcome.** Pending CI evidence at initial commit. Do not manufacture a numerical
result. Finite validation loss is a smoke-test milestone, not a strength claim.

**Known methodological limitations.** Random playouts are not representative
of human/engine games; positions sampled along games remain statistically
dependent; exact-FEN dedup does not prevent strategic overlap; no external
teacher labels; no confidence intervals, game-level ELO, ablations, or timing
benchmarks yet. There are no claims about theorem transfer in S001.
