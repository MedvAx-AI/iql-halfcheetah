"""Configuration, offline training, checkpoint evaluation and seed summaries."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .config import ProjectConfig, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IQL HalfCheetah project")
    parser.add_argument("command", choices=("check", "train", "evaluate", "summarize"))
    parser.add_argument("--config", help="Flat TOML config; omit for built-in defaults")
    parser.add_argument("--run-dir", type=Path, help="Results directory, normally results/<run_id>")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint to resume. Only total_steps and checkpoint_interval may change",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        help="Permitted resume override of config.total_steps",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow Minari to download the configured dataset before training",
    )
    parser.add_argument("--checkpoint", type=Path, help="Trusted project checkpoint to evaluate")
    parser.add_argument("--eval-episodes", type=int, help="Override checkpoint evaluation count")
    parser.add_argument("--eval-seed", type=int, help="First fixed evaluation seed")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        help="Evaluation device (default cpu); training uses its config",
    )
    parser.add_argument("--training-log", type=Path, help="Optional JSONL for training curves")
    parser.add_argument("--video", action="store_true", help="Record one seeded trained-agent MP4")
    parser.add_argument("--video-dir", type=Path, help="Default: videos/<run_id>")
    parser.add_argument("--run-dirs", type=Path, nargs="+", help="Evaluated runs to summarize")
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            if args.checkpoint is None:
                raise ValueError("evaluate requires --checkpoint")
            if args.config is not None or args.resume is not None or args.total_steps is not None:
                raise ValueError(
                    "Evaluation uses checkpoint config; use --eval-episodes/--eval-seed"
                )
            if args.video_dir is not None and not args.video:
                raise ValueError("--video-dir requires --video")
            from .reporting import evaluate_checkpoint

            run_dir = args.run_dir
            if run_dir is None:
                run_id = args.checkpoint.parent.name
                if run_id == "checkpoints":
                    run_id = args.checkpoint.parent.parent.name
                run_dir = Path("results") / run_id
            video_dir = args.video_dir or Path("videos") / run_dir.name
            result = evaluate_checkpoint(
                args.checkpoint,
                run_dir=run_dir,
                eval_episodes=args.eval_episodes,
                eval_seed=args.eval_seed,
                device=args.device or "cpu",
                training_log=args.training_log,
                video_dir=video_dir if args.video else None,
            )
            print(result)
            return 0
        if args.device is not None:
            raise ValueError("--device is an evaluation option; training uses config.device")
        if args.command == "summarize":
            if args.run_dirs is None or args.run_dir is None:
                raise ValueError("summarize requires --run-dirs and --run-dir")
            from .reporting import summarize_runs

            print(summarize_runs(args.run_dirs, output_dir=args.run_dir))
            return 0
        config = load_config(args.config) if args.config else ProjectConfig()
        if args.total_steps is not None:
            config = replace(config, total_steps=args.total_steps)
        if args.command == "check":
            print(
                json.dumps(
                    {
                        "status": (
                            "configuration valid; IQL/training and seeded evaluation/video "
                            "are implemented; performance requires measured artifacts"
                        ),
                        "config": asdict(config),
                        "owners": {
                            "integration/Colab": "Aleksandr Medvedev",
                            "dataset/environment": "Nikita Shankin",
                            "IQL/train": "Leo Vesin",
                            "evaluation/video": "Ruslan Nasibullin",
                            "notebook/slides": "Telman Nuruzov",
                        },
                    },
                    indent=2,
                )
            )
            return 0
        from .train import train

        run_dir = args.run_dir
        if run_dir is None:
            run_dir = Path(config.output_dir) / f"halfcheetah_seed_{config.seed}"
        checkpoint = train(config, run_dir=run_dir, resume=args.resume, download=args.download)
        print(checkpoint)
        return 0
    except (OSError, ValueError, NotImplementedError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
