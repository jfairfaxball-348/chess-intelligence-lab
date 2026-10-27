# S002 handoff — real Stockfish teacher corpus and first strength test

S001 delivers a synthetic pipeline, material and neural evaluators, synchronous
depth-limited alpha-beta search, optional Stockfish annotations, and a UCI
interface. The demonstration is not evidence of a strong neural engine.

## First local step (Windows laptop)

Collect an honest hardware inventory before picking training batch sizes:

    Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
    Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
    Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM
    Get-PSDrive -PSProvider FileSystem | Select-Object Name,Free,Used
    nvidia-smi

It is fine if nvidia-smi is unavailable. Record Python version, installed
Stockfish executable version, PyTorch build and whether CUDA is available.

## Session plan

1. Validate the CI green build and reproduce S001 smoke in local environment.
2. Install a native Stockfish binary independently from this repository; log its
   release identifier, options, nodes/depth and binary hash.
3. Acquire/generate legally sourced PGNs or FENs with provenance and licences.
   Use train/validation/test partitions by *source game* rather than random
   positions alone, deduplicating positions and checking mirrored positions.
4. Annotate small pilot train/test corpora with the optional UCI adapter:

       python -m chesslab.cli annotate --input data/pilot.fens --output data/teacher.jsonl --engine "C:\path\to\stockfish.exe" --depth 8 --threads 1

5. Add dataset validator, opening diversity, holdout protocol, manifest hashes,
   and experiment comparison against material-only and teacher targets.
6. Train an honest teacher-distilled scalar model (no Stockfish inference), freeze
   its seed, version and checkpoint.
7. Benchmark the material and neural engines at *equal depth and wall-clock
   budgets*, with paired openings and colours, and confidence intervals.
8. Only after reproducible real-data gains explore better features,
   policy targets, more efficient search, self-play, reinforcement learning
   and authorised formal-theorem inputs.

## Gates for S002 completion

Teacher dataset provenance is recorded; no train/test game leakage; engine
annotation is reproducible; independently reloadable model checkpoint;
paired legal UCI games and PGNs; engine outcomes and runtimes reported without
overclaim. Model quality and engine strength are different questions.

Future theorem adapter MUST consume manually approved immutable artifacts from
engame-chess and cannot claim global KRK victory status. The Maths Endgame UI
must never use this experimental engine as proof-derived play.

No changes to engame-chess or engame-chess-playground are authorised.
