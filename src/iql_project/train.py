"""Offline IQL training. Evaluation during training belongs to Person 4.

``run_dir`` is the results directory, normally ``results/<run_id>``. Checkpoints
are written to ``checkpoints/<run_id>/step_<update>.pt`` when ``run_dir`` lives
under a ``results`` directory, and to ``run_dir/checkpoints`` otherwise.
``eval_interval`` is stored in the manifest and is not used to roll out the
environment; Person 4 owns that loop.
"""

import hashlib
import json
import platform
import random
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ProjectConfig
from .dataset import load_offline_dataset
from .iql import IQLAgent, config_from_checkpoint, runtime_versions, source_commit

# Continuation controls. Every other ProjectConfig field must match the checkpoint.
# learning_rate is not overridable: Adam restores it from optimizer state, so a new
# value would be written into the manifest while the update still uses the old rate.
RESUME_OVERRIDES = frozenset({"checkpoint_interval", "total_steps"})


def train(
    config: ProjectConfig,
    *,
    run_dir: Path,
    resume: Path | None = None,
    download: bool = False,
) -> Path:
    """Train on the fixed offline dataset and return the final checkpoint path.

    On resume, only ``total_steps`` and ``checkpoint_interval`` may differ from
    the checkpoint. Algorithm settings, including ``learning_rate``, must match
    the optimizer state that ``load`` restores.
    """
    run_dir = Path(run_dir)
    resume_path = Path(resume) if resume is not None else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    resume_changes = (
        None
        if resume_path is None
        else resume_config_overrides(config, config_from_checkpoint(resume_path))
    )
    run_id = run_dir.name
    if not run_id or run_id in {".", ".."}:
        raise ValueError("run_dir must end with a run id")
    dataset = load_offline_dataset(config, download=download)
    if len(dataset) < 1:
        raise ValueError("Offline dataset is empty")
    checkpoint_dir = _checkpoint_dir(run_dir, run_id)
    log_path = run_dir / "training.jsonl"
    if resume_path is None and log_path.exists():
        raise FileExistsError(f"{log_path} already exists; pass resume to continue this run")

    if resume_path is None:
        _seed_everything(config.seed)
        agent = IQLAgent(config, dataset.info)
        sampler = np.random.default_rng(config.seed)
        agent.bind_sampler(sampler)
    else:
        agent = IQLAgent(config, dataset.info)
        sampler = np.random.default_rng(config.seed)
        agent.bind_sampler(sampler)
        agent.load(resume_path)
        if agent.step < 0 or agent.step > config.total_steps:
            raise ValueError(
                f"Checkpoint step {agent.step} is outside 0..total_steps ({config.total_steps})"
            )
    if agent.step == config.total_steps:
        raise ValueError(
            f"Checkpoint step {agent.step} has already reached total_steps; "
            "increase total_steps to resume"
        )
    if resume_path is not None and log_path.exists():
        last_step = _last_logged_step(log_path)
        if last_step != agent.step:
            raise ValueError(
                f"training.jsonl ends at step {last_step}, "
                f"but the checkpoint is at step {agent.step}"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    elapsed_offset = agent.elapsed_seconds
    final_checkpoint: Path | None = None
    with log_path.open(
        "a" if resume_path is not None and log_path.exists() else "w", encoding="utf-8"
    ) as log:
        while agent.step < config.total_steps:
            metrics = agent.update(dataset.sample(config.batch_size, sampler))
            elapsed = elapsed_offset + (time.perf_counter() - started)
            agent.elapsed_seconds = elapsed
            record = {"step": agent.step, **metrics, "elapsed_seconds": elapsed}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            if agent.step % config.checkpoint_interval == 0 or agent.step == config.total_steps:
                final_checkpoint = checkpoint_dir / f"step_{agent.step}.pt"
                agent.save(final_checkpoint)
    if final_checkpoint is None:
        raise RuntimeError("Training finished without a checkpoint")
    _write_manifest(
        run_dir=run_dir,
        run_id=run_id,
        config=config,
        dataset=dataset,
        agent=agent,
        log_path=log_path,
        checkpoint_dir=checkpoint_dir,
        final_checkpoint=final_checkpoint,
        resume_path=resume_path,
        resume_overrides=resume_changes,
        invocation_seconds=time.perf_counter() - started,
    )
    return final_checkpoint


def resume_config_overrides(
    requested: ProjectConfig, checkpoint: ProjectConfig
) -> dict[str, dict[str, Any]]:
    """Return permitted resume edits. Reject every other difference.

    The returned mapping is ``{field: {"checkpoint": old, "requested": new}}``
    for fields in ``RESUME_OVERRIDES`` that actually changed.
    """
    overrides: dict[str, dict[str, Any]] = {}
    rejected: list[str] = []
    for field in fields(ProjectConfig):
        saved = getattr(checkpoint, field.name)
        current = getattr(requested, field.name)
        if current == saved:
            continue
        if field.name in RESUME_OVERRIDES:
            overrides[field.name] = {"checkpoint": saved, "requested": current}
        else:
            rejected.append(f"{field.name} (checkpoint {saved}, requested {current})")
    if rejected:
        allowed = ", ".join(sorted(RESUME_OVERRIDES))
        changes = "; ".join(rejected)
        raise ValueError(
            f"Resume config must match the checkpoint except for {allowed}. "
            f"Rejected changes: {changes}"
        )
    return overrides


def _optimizer_learning_rates(agent: IQLAgent) -> dict[str, float]:
    return {
        "q": float(agent.q_optimizer.param_groups[0]["lr"]),
        "v": float(agent.v_optimizer.param_groups[0]["lr"]),
        "policy": float(agent.policy_optimizer.param_groups[0]["lr"]),
    }


def _checkpoint_dir(run_dir: Path, run_id: str) -> Path:
    if run_dir.parent.name == "results":
        return run_dir.parent.parent / "checkpoints" / run_id
    return run_dir / "checkpoints"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _last_logged_step(path: Path) -> int | None:
    last = None
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                last = int(json.loads(line)["step"])
    return last


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _artifact(path: Path, role: str) -> dict[str, str]:
    return {"role": role, "path": _display_path(path), "sha256": _sha256(path)}


def _write_manifest(
    *,
    run_dir: Path,
    run_id: str,
    config: ProjectConfig,
    dataset: Any,
    agent: IQLAgent,
    log_path: Path,
    checkpoint_dir: Path,
    final_checkpoint: Path,
    resume_path: Path | None,
    resume_overrides: dict[str, dict[str, Any]] | None,
    invocation_seconds: float,
) -> None:
    metadata = dataset.metadata
    preprocessing = metadata.get("preprocessing", {})
    finite_preprocessing = all(
        np.isfinite(np.asarray(value)).all()
        for value in (
            dataset.info.observation_mean,
            dataset.info.observation_std,
            dataset.info.action_low,
            dataset.info.action_high,
        )
    )
    optimizer_learning_rates = _optimizer_learning_rates(agent)
    mismatched = [
        f"{name}={rate}"
        for name, rate in optimizer_learning_rates.items()
        if rate != config.learning_rate
    ]
    if mismatched:
        raise RuntimeError(
            "Refusing to record learning_rate "
            f"{config.learning_rate}; optimizers are using {', '.join(mismatched)}"
        )
    artifacts = [_artifact(log_path, "training_log")]
    artifacts.extend(
        _artifact(path, "checkpoint") for path in sorted(checkpoint_dir.glob("step_*.pt"))
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "git_commit": source_commit(),
        "config": asdict(config),
        "dataset_id": dataset.info.dataset_id,
        "dataset_metadata": metadata,
        "dataset_cardinality": {
            "transitions": len(dataset),
            "episodes": metadata.get("total_episodes"),
        },
        "finite_preprocessing": bool(finite_preprocessing),
        "recovered_env_spec": metadata.get("env_spec"),
        "transforms": preprocessing,
        "package_versions": runtime_versions(),
        "python_version": platform.python_version(),
        "training_seed": config.seed,
        "evaluation_seeds": [config.eval_seed + index for index in range(config.eval_episodes)],
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "device": str(agent.device),
            "cuda_available": torch.cuda.is_available(),
        },
        "elapsed_seconds": agent.elapsed_seconds,
        "invocation_seconds": invocation_seconds,
        "updates_completed": agent.step,
        "optimizer_learning_rates": optimizer_learning_rates,
        "resume_overrides": resume_overrides,
        "resumed_from": None if resume_path is None else _display_path(resume_path),
        "final_checkpoint": _display_path(final_checkpoint),
        "artifacts": artifacts,
        "limitations": [
            "eval_interval is recorded but training does not roll out the environment",
            "reward_scale and reward_shift are stored; the loader leaves rewards unchanged",
        ],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
