"""Person 4: run the same offline protocol for three seeds, then evaluate checkpoints."""

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from iql_project.config import load_config
from iql_project.reporting import evaluate_checkpoint, summarize_runs
from iql_project.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/halfcheetah.toml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, help="Default: config.total_steps")
    parser.add_argument("--checkpoint-every", type=int, help="Default: config.eval_interval")
    parser.add_argument("--run-prefix", default="person4")
    parser.add_argument("--datasets-path", type=Path, default=Path("data/minari"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--video", action="store_true", help="One final video per training seed")
    args = parser.parse_args()
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use at least three distinct training seeds")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if Path(args.run_prefix).name != args.run_prefix or args.run_prefix in {".", ".."}:
        parser.error("--run-prefix must be one directory name")
    os.environ["MINARI_DATASETS_PATH"] = str(args.datasets_path.resolve())
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    if config.eval_episodes < 10:
        parser.error("Experiment protocol requires at least 10 evaluation episodes")
    config = replace(
        config,
        total_steps=args.steps if args.steps is not None else config.total_steps,
        checkpoint_interval=args.checkpoint_every
        if args.checkpoint_every is not None
        else config.eval_interval,
    )
    root = Path(config.output_dir)
    run_dirs = [root / f"{args.run_prefix}_seed_{seed}" for seed in args.seeds]
    if any(path.exists() for path in run_dirs):
        parser.error("Run directories already exist; use a fresh prefix or train --resume")
    failures = []
    for seed, run_dir in zip(args.seeds, run_dirs, strict=True):
        started = time.perf_counter()
        try:
            final = train(replace(config, seed=seed), run_dir=run_dir, download=args.download)
            checkpoints = sorted(
                final.parent.glob("step_*.pt"), key=lambda path: int(path.stem[5:])
            )
            for checkpoint in checkpoints:
                summary = evaluate_checkpoint(
                    checkpoint,
                    run_dir=run_dir,
                    video_dir=Path("videos") / run_dir.name
                    if args.video and checkpoint == final
                    else None,
                )
                print(f"seed={seed} checkpoint={checkpoint.name} summary={summary}", flush=True)
        except Exception as error:
            failures.append(
                {
                    "training_seed": seed,
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(f"seed={seed} failed: {error}", flush=True)
    if failures:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{args.run_prefix}_failures.json"
        path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"Failed runs recorded in {path}; no complete result is claimed")
    print(summarize_runs(run_dirs, output_dir=root / f"{args.run_prefix}_aggregate"))


if __name__ == "__main__":
    main()
