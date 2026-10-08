"""Checkpoint evaluation, raw artifacts and separate within/across-run summaries."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import torch

from .contracts import DatasetInfo, EvaluationResult
from .environment import make_evaluation_env
from .evaluate import RandomPolicy, evaluate, record_video
from .iql import IQLAgent, config_from_checkpoint, dataset_info_from_checkpoint, source_commit

CSV_FIELDS = (
    "run_id",
    "checkpoint_step",
    "training_seed",
    "evaluation_seed",
    "episode_index",
    "episode_return",
    "episode_length",
    "policy_name",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def artifact(path: Path, role: str) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    try:
        display = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        display = path.name
    return {"role": role, "path": display, "sha256": digest, "bytes": path.stat().st_size}


def _source_evidence() -> dict:
    package_dir = Path(__file__).resolve().parent
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=package_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except OSError:
        dirty = None
    return {
        "source_has_local_changes": dirty,
        "source_files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package_dir.glob("*.py"))
        },
    }


def _transforms(info: DatasetInfo) -> dict:
    return {
        field.name: (
            getattr(info, field.name).tolist()
            if isinstance(getattr(info, field.name), np.ndarray)
            else getattr(info, field.name)
        )
        for field in fields(info)
    }


def _rows(result: EvaluationResult, run_id: str, step: int, seed: int, policy: str) -> list[dict]:
    return [
        dict(
            zip(
                CSV_FIELDS,
                (run_id, step, seed, eval_seed, index, reward, length, policy),
                strict=True,
            )
        )
        for index, (reward, length, eval_seed) in enumerate(
            zip(result.episode_returns, result.episode_lengths, result.episode_seeds, strict=True)
        )
    ]


def read_evaluation(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError("Evaluation CSV header differs from the artifact contract")
        result = list(reader)
    for row in result:
        for name in (
            "checkpoint_step",
            "training_seed",
            "evaluation_seed",
            "episode_index",
            "episode_length",
        ):
            row[name] = int(row[name])
        row["episode_return"] = float(row["episode_return"])
        if not np.isfinite(row["episode_return"]):
            raise ValueError("Evaluation CSV contains non-finite returns")
    return result


def _validate_rows(rows: list[dict], run_id: str, protocol: dict) -> None:
    if not rows:
        raise ValueError("Evaluation CSV is empty")
    groups: dict[tuple, list[int]] = {}
    for row in rows:
        if (
            row["run_id"] != run_id
            or row["training_seed"] != protocol["training_seed"]
            or row["policy_name"] not in {"iql", "random"}
            or row["checkpoint_step"] < 0
            or row["episode_length"] <= 0
        ):
            raise ValueError("Evaluation row identity, step, length or policy is invalid")
        key = (row["policy_name"], row["checkpoint_step"])
        groups.setdefault(key, []).append(row["evaluation_seed"])
    expected = sorted(protocol["evaluation_seeds"])
    if any(sorted(seeds) != expected for seeds in groups.values()):
        raise ValueError("Each checkpoint must contain exactly the fixed evaluation seed set")
    if ("random", 0) not in groups or not any(key[0] == "iql" for key in groups):
        raise ValueError("Evaluation CSV must contain the random baseline and an IQL checkpoint")


def within_run(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["run_id"], row["training_seed"], row["checkpoint_step"], row["policy_name"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (run_id, seed, step, policy), episodes in sorted(groups.items()):
        returns = np.asarray([row["episode_return"] for row in episodes], dtype=np.float64)
        summaries.append(
            {
                "run_id": run_id,
                "training_seed": seed,
                "checkpoint_step": step,
                "policy_name": policy,
                "episodes": len(episodes),
                "evaluation_seeds": [row["evaluation_seed"] for row in episodes],
                "return_mean": float(returns.mean()),
                "return_std_population": float(returns.std()),
                "return_min": float(returns.min()),
                "return_max": float(returns.max()),
                "episode_length_mean": float(np.mean([row["episode_length"] for row in episodes])),
            }
        )
    return summaries


def _plot(run_dir: Path, rows: list[dict], training_log: Path | None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    paths = []
    if training_log is not None:
        with training_log.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if records:
            fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
            # Average consecutive blocks for legibility; raw per-update data stays in JSONL.
            block = max(1, len(records) // 1000)
            chunks = [records[i : i + block] for i in range(0, len(records), block)]
            for ax, name in zip(axes, ("q_loss", "v_loss", "policy_loss"), strict=True):
                values = np.asarray([record[name] for record in records])
                if not np.isfinite(values).all():
                    plt.close(fig)
                    raise ValueError(f"Training log has non-finite {name}")
                ax.plot(
                    [chunk[-1]["step"] for chunk in chunks],
                    [np.mean([item[name] for item in chunk]) for chunk in chunks],
                )
                ax.set_ylabel(name)
                ax.grid(alpha=0.25)
            axes[-1].set_xlabel("Offline updates")
            fig.suptitle(f"Training losses (block mean, {block} updates)")
            fig.tight_layout()
            path = plot_dir / "training_losses.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    summaries = within_run(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    for run_id in sorted({row["run_id"] for row in summaries}):
        points = [
            row for row in summaries if row["run_id"] == run_id and row["policy_name"] == "iql"
        ]
        if points:
            ax.errorbar(
                [row["checkpoint_step"] for row in points],
                [row["return_mean"] for row in points],
                yerr=[row["return_std_population"] for row in points],
                marker="o",
                capsize=3,
                label=f"{run_id}: IQL mean ± episode std",
            )
    baselines = [row for row in summaries if row["policy_name"] == "random"]
    if baselines:
        ax.axhline(
            baselines[0]["return_mean"],
            color="gray",
            linestyle="--",
            label="Uniform random mean (same evaluation seeds)",
        )
    ax.set(xlabel="Offline updates", ylabel="Raw episodic return")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = plot_dir / "evaluation_returns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return paths + [path]


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    run_dir: Path,
    eval_episodes: int | None = None,
    eval_seed: int | None = None,
    device: str = "cpu",
    training_log: Path | None = None,
    video_dir: Path | None = None,
) -> Path:
    """Evaluate one trusted project checkpoint; upsert metrics without duplicate episodes.

    Device/evaluation controls can differ; algorithm settings and preprocessing come
    from the checkpoint. Existing training manifest fields are retained.
    """
    checkpoint, run_dir = Path(checkpoint), Path(run_dir)
    saved_config = config_from_checkpoint(checkpoint)
    config = replace(
        saved_config,
        device=device,
        eval_episodes=eval_episodes if eval_episodes is not None else saved_config.eval_episodes,
        eval_seed=eval_seed if eval_seed is not None else saved_config.eval_seed,
    )
    info = dataset_info_from_checkpoint(checkpoint)
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    protocol = {
        "evaluation_seeds": [config.eval_seed + i for i in range(config.eval_episodes)],
        "dataset_id": config.dataset_id,
        "transforms": _transforms(info),
        "training_seed": config.seed,
        "training_config": asdict(saved_config),
    }
    # total_steps/checkpoint_interval can change legitimately when resuming training.
    for name in ("total_steps", "checkpoint_interval"):
        protocol["training_config"].pop(name)
    previous = manifest.get("evaluation_protocol")
    if previous is not None and previous != protocol:
        raise ValueError("Existing run uses different preprocessing/config/evaluation seeds")
    if manifest and manifest.get("training_seed", config.seed) != config.seed:
        raise ValueError("Checkpoint training seed differs from run manifest")
    if training_log is None and (run_dir / "training.jsonl").exists():
        training_log = run_dir / "training.jsonl"
    if training_log is not None and not training_log.is_file():
        raise FileNotFoundError(f"Training log not found: {training_log}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        # Agent.load restores training RNGs; evaluation must not disturb the caller.
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                agent = IQLAgent(config, info)
                agent.load(checkpoint)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        env = make_evaluation_env(config)
        try:
            env_spec = json.loads(env.spec.to_json())
        finally:
            env.close()
        if previous is not None and manifest.get("recovered_env_spec") != env_spec:
            raise ValueError("Recovered environment differs from previously evaluated checkpoints")
        result = evaluate(agent, config, info)
        baseline = evaluate(RandomPolicy(info), config, info)
        rows = _rows(result, run_dir.name, agent.step, config.seed, "iql")
        rows += _rows(baseline, run_dir.name, 0, config.seed, "random")
        csv_path = run_dir / "evaluation.csv"
        existing = read_evaluation(csv_path) if csv_path.exists() else []
        if existing:
            _validate_rows(existing, run_dir.name, protocol)
        keys = {(row["policy_name"], row["checkpoint_step"]) for row in rows}
        rows = [
            row for row in existing if (row["policy_name"], row["checkpoint_step"]) not in keys
        ] + rows
        rows.sort(
            key=lambda row: (row["policy_name"], row["checkpoint_step"], row["episode_index"])
        )
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summary_path = run_dir / "summary.json"
        _write_json(
            summary_path, {"schema_version": 1, "raw_rewards": True, "within_run": within_run(rows)}
        )
        outputs = [artifact(csv_path, "evaluation_csv"), artifact(summary_path, "summary")]
        outputs += [artifact(path, "plot") for path in _plot(run_dir, rows, training_log)]
        if video_dir is not None:
            video = record_video(agent, config, info, output_dir=video_dir, seed=config.eval_seed)
            outputs.append(artifact(video, "video"))
        manifest.setdefault("schema_version", 1)
        manifest.setdefault("run_id", run_dir.name)
        manifest.setdefault("config", asdict(saved_config))
        manifest.setdefault("git_commit", source_commit())
        manifest.setdefault("training_seed", config.seed)
        manifest.setdefault("dataset_id", config.dataset_id)
        manifest.setdefault("transforms", _transforms(info))
        manifest.setdefault("elapsed_seconds", agent.elapsed_seconds)
        manifest["recovered_env_spec"] = env_spec
        manifest["evaluation_seeds"] = protocol["evaluation_seeds"]
        manifest["evaluation_protocol"] = protocol
        manifest["evaluation_runtime"] = {
            "git_commit": source_commit(),
            **_source_evidence(),
            "python_version": platform.python_version(),
            "package_versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "gymnasium", "mujoco", "minari", "moviepy")
            },
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "device": str(agent.device),
                "torch_threads": torch.get_num_threads(),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        for name in ("python_version", "package_versions", "hardware"):
            manifest.setdefault(name, manifest["evaluation_runtime"][name])
        manifest.setdefault("evaluation_checkpoints", {})[str(agent.step)] = artifact(
            checkpoint, "checkpoint"
        )
        changed_paths = {item["path"] for item in outputs}
        manifest["artifacts"] = [
            item for item in manifest.get("artifacts", []) if item["path"] not in changed_paths
        ] + outputs
        _write_json(manifest_path, manifest)
        return summary_path
    except Exception as error:
        manifest.setdefault("failed_evaluations", []).append(
            {
                "checkpoint": checkpoint.name,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        _write_json(manifest_path, manifest)
        raise


def summarize_runs(run_dirs: list[Path], *, output_dir: Path) -> Path:
    """Compare runs at matching checkpoint steps; never pool episodes as training seeds."""
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    manifests = [_read_json(Path(path) / "manifest.json") for path in run_dirs]
    protocols = [manifest["evaluation_protocol"] for manifest in manifests]
    reference = {key: value for key, value in protocols[0].items() if key != "training_seed"}
    reference["training_config"] = {
        key: value for key, value in reference["training_config"].items() if key != "seed"
    }
    for protocol, manifest in zip(protocols, manifests, strict=True):
        other = {key: value for key, value in protocol.items() if key != "training_seed"}
        other["training_config"] = {
            key: value for key, value in other["training_config"].items() if key != "seed"
        }
        if (
            other != reference
            or manifest["recovered_env_spec"] != manifests[0]["recovered_env_spec"]
        ):
            raise ValueError(
                "Runs must share environment, preprocessing, config and evaluation seeds"
            )
    seeds = [protocol["training_seed"] for protocol in protocols]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Across-run statistics require distinct training seeds")
    run_ids = [manifest["run_id"] for manifest in manifests]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Run IDs must be distinct")
    rows = []
    for path, manifest, protocol in zip(run_dirs, manifests, protocols, strict=True):
        run_rows = read_evaluation(Path(path) / "evaluation.csv")
        _validate_rows(run_rows, manifest["run_id"], protocol)
        rows.extend(run_rows)
    summaries = within_run(rows)
    across = []
    for step in sorted(
        {row["checkpoint_step"] for row in summaries if row["policy_name"] == "iql"}
    ):
        group = [
            row
            for row in summaries
            if row["checkpoint_step"] == step and row["policy_name"] == "iql"
        ]
        means = np.asarray([row["return_mean"] for row in group])
        across.append(
            {
                "checkpoint_step": step,
                "training_seeds": [row["training_seed"] for row in group],
                "runs": len(group),
                "all_runs_present": len(group) == len(run_dirs),
                "mean_of_run_means": float(means.mean()),
                "std_sample_of_run_means": float(means.std(ddof=1)) if len(group) > 1 else None,
            }
        )
    final_steps = [
        max(
            row["checkpoint_step"]
            for row in summaries
            if row["run_id"] == run_id and row["policy_name"] == "iql"
        )
        for run_id in run_ids
    ]
    complete = (
        len(seeds) >= 3 and len(set(final_steps)) == 1 and len(reference["evaluation_seeds"]) >= 10
    )
    output_dir = Path(output_dir)
    if output_dir.resolve() in {Path(path).resolve() for path in run_dirs}:
        raise ValueError("Aggregate output directory must differ from input runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "raw_rewards": True,
            "minimum_seed_episode_protocol_met": complete,
            "within_run": summaries,
            "across_runs": across,
            "final_checkpoint_steps": dict(zip(run_ids, final_steps, strict=True)),
        },
    )
    plots = _plot(output_dir, rows, None)
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": output_dir.name,
            "git_commit": source_commit(),
            "evaluation_protocol": reference,
            "training_seeds": seeds,
            "recovered_env_spec": manifests[0]["recovered_env_spec"],
            "artifacts": [
                artifact(summary_path, "summary"),
                artifact(output_dir / "evaluation.csv", "evaluation_csv"),
            ]
            + [artifact(path, "plot") for path in plots],
            "input_manifests": [
                artifact(Path(path) / "manifest.json", "input_manifest") for path in run_dirs
            ],
        },
    )
    return summary_path
