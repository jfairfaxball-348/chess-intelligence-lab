"""Compact experiment CLI; teacher is never invoked by UCI at inference."""

import argparse
import json
from pathlib import Path

from .data import generate_split


def _print(obj: dict) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chesslab")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="Generate synthetic data, train and evaluate")
    smoke.add_argument("--out", default="runs/s001-smoke")
    smoke.add_argument("--train-count", type=int, default=96)
    smoke.add_argument("--val-count", type=int, default=24)
    smoke.add_argument("--epochs", type=int, default=3)
    smoke.add_argument("--seed", type=int, default=7)
    train = sub.add_parser("train", help="Train on an existing labelled JSONL split")
    train.add_argument("--train", required=True)
    train.add_argument("--val", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--seed", type=int, default=7)
    evaluate = sub.add_parser("evaluate", help="Compare independent model against material labels")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--dataset", required=True)
    teacher = sub.add_parser("annotate", help="Generate true UCI Stockfish labels (optional)")
    teacher.add_argument("--input", required=True)
    teacher.add_argument("--output", required=True)
    teacher.add_argument("--engine", required=True)
    teacher.add_argument("--depth", type=int, default=8)
    teacher.add_argument("--threads", type=int, default=1)
    teacher.add_argument("--hash-mb", type=int, default=32)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        from .train import train as fit, evaluate as eval_model
        root = Path(args.out)
        train_file, val_file = root / "train.jsonl", root / "val.jsonl"
        manifest = generate_split(train_file, val_file, args.train_count, args.val_count, args.seed)
        checkpoint = root / "tinyconv.pt"
        result = fit(train_file, val_file, checkpoint, epochs=args.epochs, seed=args.seed)
        report = {"dataset": manifest, "training": result,
                  "evaluation": eval_model(checkpoint, val_file)}
        root.mkdir(parents=True, exist_ok=True)
        (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
        _print(report)
    elif args.command == "train":
        from .train import train as fit
        _print(fit(args.train, args.val, args.checkpoint, epochs=args.epochs, seed=args.seed))
    elif args.command == "evaluate":
        from .train import evaluate as eval_model
        _print(eval_model(args.checkpoint, args.dataset))
    elif args.command == "annotate":
        from .teacher import annotate
        _print(annotate(args.input, args.output, args.engine,
                        depth=args.depth, threads=args.threads, hash_mb=args.hash_mb))


if __name__ == "__main__":
    main()
