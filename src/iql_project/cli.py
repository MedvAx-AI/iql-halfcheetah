"""Project commands. ``check`` validates config; ``train`` runs offline IQL."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .config import ProjectConfig, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IQL HalfCheetah project")
    parser.add_argument("command", choices=("check", "train", "evaluate"))
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
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config) if args.config else ProjectConfig()
        if args.total_steps is not None:
            config = replace(config, total_steps=args.total_steps)
        if args.command == "check":
            print(
                json.dumps(
                    {
                        "status": (
                            "configuration valid; Person 3 IQL/training is implemented; "
                            "Person 4 evaluation is not implemented"
                        ),
                        "config": asdict(config),
                        "owners": {
                            "dataset/environment": "Person 2",
                            "IQL/train": "Person 3",
                            "evaluation/video": "Person 4",
                            "notebook/slides": "Person 5",
                        },
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "evaluate":
            raise NotImplementedError("Person 4: evaluate is not implemented")
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
