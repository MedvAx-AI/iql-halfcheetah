"""Local experiment: unchanged IQL, paired baseline and continuation checkpoints."""

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from iql_project.environment import make_evaluation_env
from iql_project.iql import IQLAgent, config_from_checkpoint, dataset_info_from_checkpoint
from iql_project.reporting import evaluate_checkpoint
from iql_project.train import train


def diagnose(checkpoint, seeds):
    config = config_from_checkpoint(checkpoint)
    info = dataset_info_from_checkpoint(checkpoint)
    agent = IQLAgent(config, info)
    agent.load(checkpoint)
    env = make_evaluation_env(config)
    records = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            velocities, rewards, heights = [], [], []
            for _ in range(env.spec.max_episode_steps):
                normalized = (
                    np.asarray(observation, dtype=np.float32) - info.observation_mean
                ) / info.observation_std
                observation, reward, terminated, truncated, _ = env.step(agent.act(normalized))
                velocities.append(float(env.unwrapped.data.qvel[0]))
                heights.append(float(env.unwrapped.data.qpos[1] + 0.7))
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            velocities = np.asarray(velocities)
            stationary_tail = 0
            for velocity in velocities[::-1]:
                if abs(velocity) > 0.1:
                    break
                stationary_tail += 1
            records.append(
                {
                    "evaluation_seed": seed,
                    "episode_return": sum(rewards),
                    "episode_length": len(rewards),
                    "last_10s_mean_forward_velocity": float(velocities[-200:].mean()),
                    "stationary_tail_seconds": stationary_tail * 0.05,
                    "low_torso_fraction": float((np.asarray(heights) < 0.3).mean()),
                }
            )
    finally:
        env.close()
    return {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "step": agent.step,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "episodes": records,
        "return_mean": float(np.mean([r["episode_return"] for r in records])),
        "late_forward_velocity_mean": float(
            np.mean([r["last_10s_mean_forward_velocity"] for r in records])
        ),
        "diagnostic_scope": (
            "Descriptive movement/posture diagnostics, "
            "not the benchmark reward or a standard success criterion"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[150000, 200000, 300000, 500000])
    args = parser.parse_args()
    if mujoco.__version__ != "3.2.3":
        raise RuntimeError("Use the frozen project simulator for this comparison")
    torch.set_num_threads(1)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    baseline = diagnose(checkpoint, range(10000, 10010))
    (args.run_dir / "baseline_diagnostics.json").write_text(json.dumps(baseline, indent=2))
    evaluate_checkpoint(checkpoint, run_dir=args.run_dir)
    print("BASELINE", baseline["return_mean"], baseline["late_forward_velocity_mean"], flush=True)
    original_config = config_from_checkpoint(checkpoint)
    for target in args.steps:
        config = replace(original_config, total_steps=target, checkpoint_interval=50000)
        print("TRAINING", checkpoint.name, "to", target, flush=True)
        checkpoint = train(config, run_dir=args.run_dir, resume=checkpoint)
        evaluate_checkpoint(checkpoint, run_dir=args.run_dir)
        result = diagnose(checkpoint, range(10000, 10010))
        (args.run_dir / f"diagnostics_{target}.json").write_text(json.dumps(result, indent=2))
        print(
            "RESULT",
            target,
            result["return_mean"],
            result["late_forward_velocity_mean"],
            result["episodes"][0],
            flush=True,
        )


if __name__ == "__main__":
    main()
