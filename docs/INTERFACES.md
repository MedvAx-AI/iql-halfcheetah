# Shared interfaces (version 1)

All public boundaries are in `src/iql_project/contracts.py`. Type annotations are
contracts, not runtime validators: Person 2 must validate its loader/sampler outputs.
No imports may download data, start training or initialize a rendering context.

## Data → learner

`load_offline_dataset(config, download=False) -> OfflineDataset` exposes `.info`,
`len(dataset)` and `sample(batch_size, rng) -> TransitionBatch`. Numeric arrays are
float32; observation shapes `(B,17)`, actions `(B,6)`, rewards `(B,1)`. End flags
are bool `(B,1)`. Observations and next observations use identical training-data
normalization. Actions keep environment units/bounds; document any policy transform.

Minari episodes have T actions/rewards and T+1 observations. Pair within each episode
only. True termination disables bootstrapping; truncation alone keeps it. Episode
end is `terminated or truncated` for evaluation/control flow, never the Q target mask.

`DatasetInfo` stores dataset ID, dimensions, action bounds, observation mean/std and
reward scale/shift. Std includes the chosen epsilon convention; no recomputing
statistics from evaluation trajectories. Confirm data cardinality/finite values
and recovered EnvSpec in the run manifest (in addition to `.info`).

## Learner → evaluator

`IQLAgent(config, info)` implements the `Learner` protocol. `update(batch)` returns
finite `q_loss`, `v_loss`, `policy_loss`, `advantage_mean` floats. Tensor conversion
belongs to the agent. `act(preprocessed_observation, deterministic=True)` accepts
one `(17,)` float32 array and returns a bounded `(6,)` action. Evaluation owns raw
observation normalization using saved checkpoint statistics.

Checkpoint save/load contains model/target networks, optimizer states, step, RNG
states, complete config, dataset ID, preprocessing, versions and source commit.
Person 3 may add a loading factory, but must preserve the protocol and explain how
evaluation recovers `DatasetInfo` before constructing the agent.
`dataset_info_from_checkpoint(path)` and `config_from_checkpoint(path)` read that
metadata without building an agent. Evaluation should rebuild `DatasetInfo` with
the checkpoint helper, construct `IQLAgent(config, info)`, then call `load`.
Normalize each raw observation with the restored mean and std before `act`.
The loss, likelihood and resume conventions are in `docs/IQL.md`.

## Training and evaluation

`train(config, run_dir=Path(...)) -> Path` returns the final checkpoint; logs raw
update metrics and provenance. It trains on the fixed dataset only. Person 3 wires
the future train CLI and checkpoint/resume flags in the same PR.

`make_evaluation_env(config, render_mode=None)` recovers the dataset's exact EnvSpec.
Person 2 must avoid accidental differences in reward weights, horizon or observations.

`evaluate(policy, config, info) -> EvaluationResult` supplies per-episode return,
length and seed tuples of length `config.eval_episodes`.
`record_video(policy, config, info, output_dir=..., seed=...) -> Path` writes MP4.
Person 4 wires future evaluate CLI/checkpoint arguments. Evaluation uses saved
training preprocessing but reports untransformed environment rewards.

## Artifact layout / required fields

```text
results/<run_id>/manifest.json
results/<run_id>/training.jsonl
results/<run_id>/evaluation.csv
results/<run_id>/summary.json
results/<run_id>/plots/
checkpoints/<run_id>/step_<update>.pt
videos/<run_id>/demo_seed_<seed>.mp4
```

- Manifest: schema_version=1, run_id, git_commit, config, dataset_id, dataset metadata,
  recovered env specification, transforms, package versions, Python version,
  training_seed, evaluation_seeds, hardware, elapsed_seconds and artifact URLs/checksums.
- Training JSONL: step, q_loss, v_loss, policy_loss, advantage_mean, elapsed_seconds.
- Evaluation CSV: run_id, checkpoint_step, training_seed, evaluation_seed,
  episode_index, episode_return, episode_length, policy_name.
- Summary: raw-return statistics within each seed and across seeds; no normalized
  score unless reference returns and formula are sourced for these exact settings.

These are handoff requirements; serialization and evaluation remain role work.
