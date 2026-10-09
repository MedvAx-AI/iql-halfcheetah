"""Measure every continuation stage on one runtime; retain all fixed-seed episodes."""

import argparse
import hashlib
import json
import math
import platform
import urllib.request
import uuid
from pathlib import Path

import mujoco
import numpy as np
import torch

from iql_project.environment import make_evaluation_env
from iql_project.iql import (
    IQLAgent,
    config_from_checkpoint,
    dataset_info_from_checkpoint,
    source_commit,
)


def audit_policy(checkpoint, seeds=None):
    config = config_from_checkpoint(checkpoint)
    info = dataset_info_from_checkpoint(checkpoint)
    policy = IQLAgent(config, info)
    policy.load(checkpoint)
    env = make_evaluation_env(config)
    episodes = []
    try:
        for seed in (
            seeds if seeds is not None else list(range(10000, 10010)) + list(range(20000, 20010))
        ):
            obs, _ = env.reset(seed=seed)
            velocities, heights, angles, rewards = [], [], [], []
            for _ in range(1000):
                normalized = (
                    np.asarray(obs, dtype=np.float32) - info.observation_mean
                ) / info.observation_std
                obs, reward, terminated, truncated, _ = env.step(policy.act(normalized))
                velocities.append(float(env.unwrapped.data.qvel[0]))
                heights.append(float(env.unwrapped.data.qpos[1] + 0.7))
                angles.append(float(env.unwrapped.data.qpos[2]))
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            tail = 0
            for velocity in reversed(velocities):
                if abs(velocity) > 0.1:
                    break
                tail += 1
            episodes.append(
                {
                    "evaluation_seed": seed,
                    "episode_return": math.fsum(rewards),
                    "episode_length": len(rewards),
                    "last_10s_mean_forward_velocity": float(np.mean(velocities[-200:])),
                    "stationary_tail_seconds": tail * 0.05,
                    "low_torso_fraction": float(np.mean(np.asarray(heights) < 0.3)),
                    "inverted_fraction": float(np.mean(np.cos(angles) < 0)),
                }
            )
    finally:
        env.close()
    return {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "step": policy.step,
        "episodes": episodes,
    }


def main(original_dir, output_dir, inventory):
    torch.set_num_threads(1)
    assert mujoco.__version__ == "3.2.3"
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "source_commit": source_commit(),
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "torch": torch.__version__,
        "audit_session": uuid.uuid4().hex,
    }
    baselines = {}
    for seed in range(3):
        checkpoint = original_dir / f"person4_100k_seed_{seed}/step_100000.pt"
        baselines[f"seed_{seed}_baseline_100k"] = audit_policy(checkpoint)
    for step in (150000, 200000, 300000, 500000):
        audit = {**common, "runs": dict(baselines)}
        for seed in range(3):
            asset = inventory[f"seed_{seed}_step_{step}.pt"]
            checkpoint = output_dir / f"seed_{seed}_step_{step}.pt"
            urllib.request.urlretrieve(asset["url"], checkpoint)
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == asset["sha256"]
            result = audit_policy(checkpoint)
            assert result["step"] == step
            audit["runs"][f"seed_{seed}_candidate_{step // 1000}k"] = result
            for start in (10000, 20000):
                episodes = [
                    r for r in result["episodes"] if start <= r["evaluation_seed"] < start + 10
                ]
                print(
                    "MEASURED",
                    seed,
                    step,
                    start,
                    "mean",
                    np.mean([r["episode_return"] for r in episodes]),
                    "stationary tails >=5s",
                    sum(r["stationary_tail_seconds"] >= 5 for r in episodes),
                    flush=True,
                )
            print("FIXED DEMO", seed, step, result["episodes"][0], flush=True)
        (output_dir / f"colab-all-seeds-{step // 1000}k.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        print(f"IQL_ALL_SEED_AUDIT_{step // 1000}K_JSON " + json.dumps(audit), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    main(args.original_dir, args.output_dir, json.loads(args.inventory.read_text()))
