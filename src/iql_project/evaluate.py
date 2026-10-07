"""Seeded rollouts using checkpoint preprocessing and raw environment rewards."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .config import ProjectConfig
from .contracts import DatasetInfo, EvaluationResult, FloatArray, Policy
from .environment import make_evaluation_env


class RandomPolicy:
    """Uniform bounded baseline, with an independent stream per evaluation seed."""

    def __init__(self, info: DatasetInfo) -> None:
        self.info = info
        self.rng = np.random.default_rng(0)

    def act(self, observation: FloatArray, *, deterministic: bool = True) -> FloatArray:
        return self.rng.uniform(self.info.action_low, self.info.action_high).astype(np.float32)


def _validate_info(config: ProjectConfig, info: DatasetInfo) -> None:
    if (info.dataset_id, info.observation_dim, info.action_dim) != (
        config.dataset_id,
        config.observation_dim,
        config.action_dim,
    ):
        raise ValueError("Evaluation config differs from checkpoint dataset/dimensions")
    for name, dimension in (
        ("observation_mean", info.observation_dim),
        ("observation_std", info.observation_dim),
        ("action_low", info.action_dim),
        ("action_high", info.action_dim),
    ):
        array = np.asarray(getattr(info, name))
        if array.shape != (dimension,) or not np.isfinite(array).all():
            raise ValueError(f"Invalid checkpoint {name}")
    if np.any(info.observation_std <= 0) or np.any(info.action_low >= info.action_high):
        raise ValueError("Checkpoint std must be positive and action low < high")


def _observation(raw: object, info: DatasetInfo) -> FloatArray:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (info.observation_dim,) or not np.isfinite(raw).all():
        raise ValueError("Environment observation has invalid shape or non-finite values")
    normalized = ((raw - info.observation_mean) / info.observation_std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite normalized observation")
    return normalized


def _episode(policy: Policy, env: object, info: DatasetInfo, seed: int) -> tuple[float, int]:
    if not (
        np.array_equal(env.action_space.low, info.action_low)
        and np.array_equal(env.action_space.high, info.action_high)
    ):
        raise ValueError("Checkpoint action bounds differ from recovered environment")
    if isinstance(policy, RandomPolicy):
        policy.rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed)
    horizon = env.spec.max_episode_steps if env.spec is not None else None
    if horizon is None or horizon <= 0:
        raise ValueError("Recovered environment must have a finite positive episode horizon")
    rewards = []
    for length in range(1, horizon + 1):
        action = np.asarray(policy.act(_observation(observation, info), deterministic=True))
        if (
            action.shape != (info.action_dim,)
            or not np.isfinite(action).all()
            or np.any(action < info.action_low)
            or np.any(action > info.action_high)
        ):
            raise ValueError("Policy action has invalid shape, non-finite values or bounds")
        observation, reward, terminated, truncated, _ = env.step(action)
        _observation(observation, info)
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("Environment reward is non-finite")
        rewards.append(reward)
        if terminated or truncated:
            total = math.fsum(rewards)
            if not math.isfinite(total):
                raise ValueError("Episode return is non-finite")
            return total, length
    raise RuntimeError("Recovered environment did not end at its recorded horizon")


def evaluate(policy: Policy, config: ProjectConfig, info: DatasetInfo) -> EvaluationResult:
    """Use raw observations -> saved normalization -> policy; report raw rewards."""
    _validate_info(config, info)
    env = make_evaluation_env(config)
    try:
        seeds = tuple(config.eval_seed + index for index in range(config.eval_episodes))
        episodes = [_episode(policy, env, info, seed) for seed in seeds]
        return EvaluationResult(
            episode_returns=tuple(result[0] for result in episodes),
            episode_lengths=tuple(result[1] for result in episodes),
            episode_seeds=seeds,
        )
    finally:
        env.close()


def record_video(
    policy: Policy, config: ProjectConfig, info: DatasetInfo, *, output_dir: Path, seed: int
) -> Path:
    """Record one deterministic policy episode via rgb_array; return playable MP4 path."""
    from gymnasium.wrappers import RecordVideo

    _validate_info(config, info)
    if type(seed) is not int or seed < 0:
        raise ValueError("Video seed must be a nonnegative integer")
    output_dir = Path(output_dir)
    target = output_dir / f"demo_seed_{seed}.mp4"
    if target.exists():
        raise FileExistsError(f"Video already exists: {target}")
    env = make_evaluation_env(config, render_mode="rgb_array")
    wrapped = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".recording-", dir=output_dir) as temporary:
            recording_dir = Path(temporary) / "frames"
            wrapped = RecordVideo(
                env,
                video_folder=str(recording_dir),
                name_prefix=f"demo_seed_{seed}",
                episode_trigger=lambda episode: episode == 0,
                disable_logger=True,
            )
            try:
                _episode(policy, wrapped, info, seed)
            finally:
                wrapped.close()
            generated = recording_dir / f"demo_seed_{seed}-episode-0.mp4"
            if not generated.is_file() or generated.stat().st_size == 0:
                raise RuntimeError("Recorder did not produce an MP4")
            generated.replace(target)
    finally:
        if wrapped is None:
            env.close()
    return target
