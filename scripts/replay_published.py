"""Replay published final checkpoints twice on this platform; preserve both results."""

import argparse
import hashlib
import json
import platform
import tarfile
import urllib.request
from pathlib import Path

from iql_project.iql import runtime_versions, source_commit
from iql_project.reporting import evaluate_checkpoint, read_evaluation, summarize_runs


def verify_repeat(first: list[dict], repeated: list[dict]) -> None:
    def episodes(rows: list[dict]) -> list[dict]:
        return [{key: value for key, value in row.items() if key != "run_id"} for row in rows]

    if episodes(first) != episodes(repeated):
        raise ValueError("Repeated evaluation differs on this platform; inspect episode CSVs")


def replay(output_dir: Path, *, video: bool = False) -> Path:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "docs/evidence/person-4"
    publication = json.loads((evidence / "publication.json").read_text())
    bundle = json.loads((evidence / "bundle_manifest.json").read_text())["artifact"]
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "person4_artifacts.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(bundle["url"], archive)
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != bundle["sha256"]:
            raise ValueError("Published checkpoint archive checksum mismatch")
    checkpoints = {}
    with tarfile.open(archive, "r:gz") as source:
        for seed in range(3):
            member = f"checkpoints/person4_100k_seed_{seed}/step_100000.pt"
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing checkpoint {member}")
            payload = stream.read()
            expected = next(
                asset["checkpoint_sha256"]
                for asset in publication["assets"]
                if asset.get("variant") == "visible_floor" and asset["training_seed"] == seed
            )
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Checkpoint {seed} checksum mismatch")
            path = output_dir / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            checkpoints[seed] = {"path": path, "sha256": expected}
    runs = []
    for seed, checkpoint in checkpoints.items():
        first = output_dir / f"seed_{seed}"
        repeated = output_dir / f"repeat_seed_{seed}"
        for run in (first, repeated):
            evaluate_checkpoint(
                checkpoint["path"],
                run_dir=run,
                eval_episodes=10,
                eval_seed=10000,
                video_dir=output_dir / "video" if video and seed == 0 and run == first else None,
            )
        verify_repeat(
            read_evaluation(first / "evaluation.csv"), read_evaluation(repeated / "evaluation.csv")
        )
        runs.append(first)
        print(f"seed {seed}: all 20 IQL/baseline episode records repeat exactly", flush=True)
    summary = summarize_runs(runs, output_dir=output_dir / "aggregate")
    original = json.loads((evidence / "summary.json").read_text())
    result = {
        "schema_version": 1,
        "source_commit": source_commit(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "package_versions": runtime_versions(),
        "archive_sha256": bundle["sha256"],
        "checkpoint_sha256": {str(seed): item["sha256"] for seed, item in checkpoints.items()},
        "evaluation_seeds": list(range(10000, 10010)),
        "same_platform_exact_repeat": True,
        "episode_records_compared": 60,
        "published_platform": "Apple M3 Pro / macOS arm64 / CPU",
        "published_final": original["across_runs"][-1],
        "this_platform_final": json.loads(summary.read_text())["across_runs"][-1],
        "scope": "Exact same-platform replay; cross-platform returns are separately reported",
    }
    target = output_dir / "repeatability.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["this_platform_final"], indent=2), flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    print(replay(args.output_dir.resolve(), video=args.video))


if __name__ == "__main__":
    main()
