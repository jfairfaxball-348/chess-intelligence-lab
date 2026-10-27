# S002 - local Stockfish distillation result

S002 ran on the owner's Windows laptop on 27 October 2026. The selected neural
checkpoint improved held-out prediction error over its untrained initialization,
but did not beat material. Playing strength remains unmeasured.

## Measured result

| Evaluator | Held-out MAE, pawns |
| --- | ---: |
| Material only | 1.730000 |
| Selected tinyconv-v1 | 2.779642 |
| Untrained tinyconv-v1, seed 7 | 3.743663 |
| Constant zero | 3.736666 |

Neural RMSE was 4.159533 pawns. These are 303 finite centipawn labels from eight
held-out games. The other 15 held-out positions have explicit mate labels and
were excluded from scalar regression. Positions within a game are correlated;
303 positions are not 303 independent games. No Elo or playing-strength gain is claimed.

The neural MAE decreased by about 25.8% from initialization, but is about 60.7%
higher than material. The experiment establishes learning of some predictive
signal and a working local pipeline. It does not establish a useful replacement
for material. The tiny pooled CNN, absolute-color encoding with a turn plane,
small correlated corpus and shallow teachers are plausible limitations; this
run does not isolate their contributions. Do not tune on this now-observed test
set or launch a larger run solely because training operated successfully.

## Local environment and bounds

- Persistent checkout: `C:\chess-intelligence-lab`.
- Windows 11 Home 10.0.26300; Python 3.12.10; PyTorch 2.14.1+cpu.
- Intel i7-11800H: eight cores, 16 logical processors; 16,866,881,536 bytes RAM.
- NVIDIA RTX 3050 Ti Laptop GPU: 4,096 MiB, driver 595.97. GPU was detected;
  this deliberately small run used the existing CPU training path, one thread.
  The installed CPU wheel reports CUDA unavailable; that is not a hardware claim.
- Approximately 77 GB free before setup, 76 GB after environment installation.
- Native [Stockfish 19 official release](https://github.com/official-stockfish/Stockfish/releases/tag/sf_19),
  Windows x86-64 universal binary, GPLv3. The executable hash, exact local path,
  dependency snapshot and hardware inventory are in the adjacent manifests.
- Started from clean, fetched `main`, base commit
  `730961d331f150f1d33303dac6c7c91e9b7508af`. S002 executed with the source changes
  committed alongside this report; the result's `git_sha` is the base, not a claim
  that the working tree was clean. Numerical source hashes are recorded separately.

The original 14 tests passed locally. The S001 96/24-position synthetic smoke
also completed, with no useful learning claim. The final expanded suite has 17
passing tests. Six depth-1 UCI checks returned legal moves using material and the
reloaded neural checkpoint, without invoking the teacher at inference.

## Corpus and reproducibility

Generated 80 local Stockfish-guided games in 18.8 seconds, using depth 4, MultiPV
3, move-selection weights 8:2:1, one thread, 32 MiB hash, seed 20261027 plus game
index, and a 240-ply cap. Game-level hash is reset before each game. These are
seeded engine-guided games, not uniform random playouts and not representative
of human play. There is no third-party PGN corpus or game-source licence dependency.
The Stockfish distribution retains its licence and source information locally.

Games were shuffled and assigned before position extraction: 64 training,
eight validation, eight test. Up to 40 stratified positions per game were taken
from ply eight onward. Global exclusion uses clock-free legal-en-passant FEN,
color/rank mirroring, and file reflection only when castling rights are absent.
Test, then validation, then training receive first ownership of shared positions.
This removes recognized duplicate positions across all partitions and within each.

| Split | Positions | Finite CP | Mate labels |
| --- | ---: | ---: | ---: |
| Training | 2,520 | 2,385 | 135 |
| Validation | 316 | 298 | 18 |
| Test | 318 | 303 | 15 |

All labels use Stockfish depth 8, one thread, 32 MiB hash, with a fresh game and
cleared hash per position. Targets are from the side-to-move perspective.
Teacher identity, executable SHA-256, best move, PV, nodes and achieved depth
are retained for every row. Input/game provenance and hashes are retained.
There are opening, middlegame and endgame samples and both positive and negative
evaluations; the audit gives actual distributions rather than a representativeness claim.

The initial 64-position benchmark took 2.14 seconds, 29.91 positions/second.
Full train/validation/test annotation took 30.97 / 4.44 / 4.39 seconds.
The adapter flushes and fsyncs each row, rejects changed input/settings/engine
on resume, and repairs an incomplete final line. Mock crash/restart tests and a
real Stockfish interrupted-prefix comparison verified recovery without relabelling
completed rows. A complete malformed line fails closed.

The independent artifact audit checked legal source games, legal teacher PVs,
source hashes, finite-versus-mate consistency, depths, game separation and
canonical-position separation. All passed. Data and checkpoint hashes are in
`S002-dataset-manifest.json`, `S002-annotation.json`, and `S002-results.json`.

## Training and selection

Existing tinyconv-v1, planes19-v1, width 32, 15,841 parameters, MSE/AdamW,
batch 32, seed 7, CPU one thread. Two choices were declared in the runner before
training: 20 epochs at learning rate 0.002 and 30 epochs at 0.0005. Each saved
its best validation-MAE epoch; selection between trials also used validation only.

- Model v001: best epoch 20, validation MAE 2.822334 pawns.
- Model v002: best epoch 22, validation MAE 2.849246 pawns.

Model v001 was selected. The test set was evaluated after selection. Training
ran from 13:29:16 to 13:29:32 UTC. Both checkpoints are reloadable and preserved;
existing checkpoint paths now fail rather than overwrite silently.

## Persistence, backup and remaining gate

Raw games: `C:\chess-intelligence-lab\data\raw\engine-v001`.
Splits: `C:\chess-intelligence-lab\data\splits\engine-v001`.
Labels and enriched provenance rows:
`C:\chess-intelligence-lab\data\teacher\teacher-v001`.
Weights: `C:\chess-intelligence-lab\checkpoints\s002\model-v001.pt` and `model-v002.pt`.
Full logs, environment, smoke, audit and metrics: `C:\chess-intelligence-lab\runs\s002`.

These ordinary files are on the persistent C: volume. A written persistence
probe was reopened in another command process. The session did not reboot the
laptop; durability across reboot follows ordinary filesystem storage, not a
performed reboot test. Do not remove this checkout or ignored directories.

Only C: is mounted. The owner explicitly chose to run everything locally with
no second location. No independent backup has been made; GitHub contains only
source and small textual reports/manifests. When a separate drive is available,
copy data/checkpoints/run metadata there and verify every file by SHA-256 before
calling it a backup. This remains an acknowledged storage risk, not a completed
backup gate.

A controlled timed UCI match remains unexecuted. The current engine is synchronous,
depth-only, without clock-aware interruption or stop support. Depth parity does
not establish equal compute. Implement and validate those controls before timed
material-versus-neural games; save paired openings, colors, budgets, PGNs and
timings. No match W/D/L or Elo is inferred from the UCI smoke.

See `../S003-handoff.md` for exact local reuse commands. S003 is future work.
