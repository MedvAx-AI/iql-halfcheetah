# Person 3 — IQL, training and checkpoints

Implements [issue #2](https://github.com/MedvAx-AI/iql-halfcheetah/issues/2).
Shared `ProjectConfig` fields are unchanged. Network width is fixed below so the
TOML contract does not grow without a lead review.

## Losses

Notation matches the notebook: `expectile` is τ, `inverse_temperature` is β,
and `max_weight` caps the policy weight. Discount is γ. An update uses one
offline `TransitionBatch` and does not touch the environment.

1. **Value.** Let `Q̄(s, a) = min(target Q1, target Q2)` on the dataset action.
   The residual is `u = Q̄ - V`. The loss is the mean of `|τ - 1(u < 0)| u²`,
   so a positive residual (Q above V) has weight τ and a negative residual has
   weight `1 - τ`. `u = 0` uses τ. Target critics are frozen copies: this loss
   does not update them and does not backpropagate into the online critics.

2. **Twin Q.** Both online critics regress to the same target

   ```text
   y = reward + γ * (1 - terminated) * V(next_observation)
   ```

   `V(next)` is detached. Truncation is stored and checked, and it does not
   enter `y`. Time-limit bootstrapping stays on; evaluation must still end the
   episode when either flag is set. `q_loss = MSE(Q1, y) + MSE(Q2, y)`, so each
   critic gets the gradient of its own mean squared error. There is no target
   value network and no `max` over actions.

3. **Policy.** After the value step, `A = Q̄(s, a) - V(s)` with both sides
   detached. The weight is `min(exp(β A), max_weight)`, implemented as
   `exp(min(β A, log max_weight))` so a large advantage cannot overflow.
   The policy loss is the mean of `-stopgrad(weight) * log π(a|s)` on the
   dataset action. Gradients flow only through that likelihood.

4. **Target update.** After the three optimizer steps,
   `θ' ← (1 - target_tau) θ' + target_tau θ` for each critic. `target_tau`
   is the Polyak rate, not the expectile.

`update` returns finite Python floats `q_loss`, `v_loss`, `policy_loss` and
`advantage_mean` (the batch mean of `A` above). A non-finite metric or a
gradient that reaches a network outside the loss being optimized raises
`RuntimeError`.

## Policy likelihood

The actor is a diagonal Gaussian in pre-tanh coordinates, with state-dependent
`log σ` clamped to `[-5, 2]`. The mean head starts near zero and the log-std
bias starts at `-0.5`. Hidden layers are two ReLU layers of width 256 for Q, V
and the policy trunk.

Environment actions use dataset bounds `(action_low, action_high)`:

```text
u = 2 (a - low) / (high - low) - 1
z = atanh(u)
log π(a|s) = Σ [log N(z; μ, σ) - log(1 - u²) - log((high - low) / 2)]
```

`u` is clamped into `(-1 + 1e-6, 1 - 1e-6)` before `atanh`, because dataset
actions may lie on the bound. For HalfCheetah `[-1, 1]` the scale term is
`log 1 = 0`. `act(observation, deterministic=True)` returns `squash(μ)` and
does not consume the torch RNG. Stochastic actions sample `z` and squash it.
Both stay inside the recorded bounds. `act` expects one already normalized
observation of shape `(17,)` and returns a float32 action of shape `(6,)`.

## Checkpoint and how evaluation should load it

`IQLAgent.save` writes schema version 1: online and target networks, Adam
states, update step, elapsed seconds, the full config, dataset ID, preprocessing
(mean, std, reward scale/shift, action bounds), package versions, the source
git commit, and RNG state for PyTorch, Python `random`, global NumPy, and the
bound `numpy.random.Generator` used to sample batches.

Person 4 recovers the contract without guessing statistics:

```python
from iql_project.iql import IQLAgent, config_from_checkpoint, dataset_info_from_checkpoint

config = config_from_checkpoint(path)
info = dataset_info_from_checkpoint(path)
agent = IQLAgent(config, info)
agent.load(path)
preprocessed = (raw_observation - info.observation_mean) / info.observation_std
action = agent.act(preprocessed.astype("float32"), deterministic=True)
```

`load` rejects a different dataset ID, dimension, action bound, observation
mean/std, or reward scale/shift. Rewards in the batch are already in
environment units (`reward_scale` is 1 and `reward_shift` is 0); the agent does
not transform them again.

## Training

```bash
uv run --frozen iql-project train --config configs/halfcheetah.toml \
  --run-dir results/<run_id> --download
uv run --frozen iql-project train --config configs/halfcheetah.toml \
  --run-dir results/<run_id> --resume checkpoints/<run_id>/step_<update>.pt \
  --total-steps 500000
```

`--download` is opt-in. Without it, a missing Minari cache raises
`FileNotFoundError` before any run directory is created.

Resume restores Adam, including each parameter group's learning rate. The
requested config must therefore match the checkpoint except for two continuation
controls: `total_steps` and `checkpoint_interval`. A different `learning_rate`,
discount, expectile, temperature, batch size, seed, device or dataset id is
rejected before a manifest is written. `--total-steps` is the command-line form
of the `total_steps` override. The manifest's `config.learning_rate` matches
`optimizer_learning_rates`, and `resume_overrides` lists only the permitted
fields that changed.

`train(config, run_dir=...)` seeds Python, NumPy and PyTorch from `config.seed`
before the networks are created, then draws batches with
`np.random.default_rng(config.seed)`. Resume does not reseed: it restores the
checkpoint RNGs, optimizer state and step, then continues until `total_steps`.
Logs append when the same `run_dir` already contains `training.jsonl` ending at
the checkpoint step.

Artifacts:

```text
results/<run_id>/training.jsonl    step, q_loss, v_loss, policy_loss, advantage_mean, elapsed_seconds
results/<run_id>/manifest.json     schema 1, commit, config, dataset metadata, env spec, versions, hardware, checksums
checkpoints/<run_id>/step_<update>.pt
```

If `run_dir` is not inside a directory named `results`, checkpoints are written
to `run_dir/checkpoints` instead. `eval_interval` is copied into the manifest
and is not used to roll out HalfCheetah. That loop is Person 4's.

## Limitations

- The checked real-data run is 1,000 updates, not the default 500,000, and it
  does not report a HalfCheetah return. Numbers are in
  `docs/evidence/person-3/`.
- `device = "cuda"` requires a CUDA build of PyTorch. The committed lock is the
  CPU profile; this machine was not used as a GPU measurement.
- Hidden width, layer count and log-std clamps are code constants, not TOML keys.
- No behavior-cloning baseline, reward normalization, or in-training evaluation
  is included.
