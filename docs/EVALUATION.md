# Person 4 — evaluation, experiments and demo

Implements [issue #3](https://github.com/MedvAx-AI/iql-halfcheetah/issues/3).
The shared `ProjectConfig`, `Policy` and `EvaluationResult` signatures are unchanged.

## Evaluation protocol

`evaluate(policy, config, info)` recovers the Minari collection environment through
`make_evaluation_env`. It resets each episode with seeds
`eval_seed .. eval_seed + eval_episodes - 1`, normalizes every observation with the
checkpoint mean/std, and calls `act(..., deterministic=True)`. It sums **raw**
environment rewards, ends on **either** termination or truncation, and closes the
environment on success or error. Invalid observations, actions, preprocessing or
rewards fail clearly. No rollout is passed to the dataset or learner update.

The default comparison uses training seeds **0, 1, 2** and the same ten evaluation
seeds **10000–10009** at every checkpoint. The random baseline samples uniformly
inside the checkpoint's action bounds. Its NumPy Generator is reset for every
evaluation seed, so its action sequence does not depend on prior evaluations or a
training seed. Both policies use identical recovered environment/reward/horizon
settings. We report raw episodic returns and do not calculate a D4RL normalized score.

Statistics are deliberately separate:

- Within one checkpoint/run: episode mean, population standard deviation (`ddof=0`),
  min/max and mean episode length. These describe those evaluated episodes.
- Across training seeds at the **same update step**: mean of each run's episode mean,
  and sample standard deviation of those run means (`ddof=1`). One run has no
  across-run standard deviation; JSON stores `null`. Episodes are never pooled and
  relabeled as independent training runs.

`minimum_seed_episode_protocol_met` requires at least three distinct training seeds,
at least ten shared evaluation episodes, and identical final checkpoint steps.
This flag establishes the comparison protocol; it does not establish a performance
threshold or completion of a 500,000-update training schedule.

## Existing checkpoint

From the repository root, after the locked dev/notebook installation:

```bash
export MINARI_DATASETS_PATH="$PWD/data/minari"
uv run --frozen iql-project evaluate \
  --checkpoint checkpoints/<run_id>/step_100000.pt \
  --run-dir results/<run_id> --eval-episodes 10 --eval-seed 10000 --video
```

The command recovers config/preprocessing from a **trusted project checkpoint**.
`--device cpu` is the portable evaluation default, even for checkpoints trained
on CUDA. Training hyperparameters cannot be supplied through `--config` here.
Checkpoint construction/load temporarily preserves the caller's Python, NumPy and
PyTorch RNG states; deterministic evaluation consumes none of those training streams.

Run the same command for intermediate checkpoints with the same `--run-dir` and
without `--video`. CSV rows for an already evaluated checkpoint are replaced, so
reruns do not duplicate episodes. The random baseline is stored once at step 0.
The existing training manifest is retained and extended. A different evaluation
seed set, training seed, algorithm config or preprocessing requires another run
directory. `--training-log` supplies a JSONL elsewhere; otherwise the command uses
`<run-dir>/training.jsonl` when available.

Artifacts follow `docs/INTERFACES.md`:

```text
results/<run_id>/evaluation.csv          every return, length, seed, policy and step
results/<run_id>/summary.json            within-run statistics at each checkpoint
results/<run_id>/plots/training_losses.png
results/<run_id>/plots/evaluation_returns.png
results/<run_id>/manifest.json           config, environment, runtime, checksums, failures
videos/<run_id>/demo_seed_10000.mp4       one deterministic final-policy episode
```

The training plot shows q/v/policy losses against offline updates. For long logs,
consecutive blocks are averaged to at most roughly 1,000 plotted points; its title
states the block size. Evaluation error bars show the within-run episode standard
deviation. The CSV/JSONL retain unsmoothed data.

Video uses Gymnasium `RecordVideo`, `rgb_array`, the recovered render FPS, and the
same observation preprocessing/action policy as evaluation. The selected seed is
fixed in advance, rather than selecting the best rollout. The encoder writes into
a temporary subdirectory, and the final MP4 appears only after a complete episode
and successful encoding. An existing final filename is protected from overwrite.
`gymnasium[other]` in the locked environment already supplies MoviePy/FFmpeg; no
dependency changes are needed. Linux headless rendering may require `MUJOCO_GL=egl`
and graphics-driver libraries; macOS uses its native OpenGL context.

## Three-seed experiment

The script calls Person 3's public `train` API, saves intermediate checkpoints,
then evaluates them offline. It performs no online training or rollout-buffer writes.

```bash
uv run --frozen python scripts/run_experiments.py \
  --steps 100000 --checkpoint-every 25000 --run-prefix person4_100k --video --download
```

Download is explicit; omit `--download` once the dataset exists. Defaults are three
seeds, ten evaluation episodes, one CPU PyTorch thread, and the shared TOML's total
steps/checkpoint spacing. A fresh run prefix protects prior results. Failures retain
their error and elapsed time in `results/<prefix>_failures.json`; evaluation failures
also go into the run manifest. A failed experiment does not emit a complete aggregate.
Resume training with Person 3's `train --resume` command, then evaluate the remaining
checkpoints explicitly. The experiment script itself does not resume existing runs.

```bash
uv run --frozen iql-project summarize \
  --run-dirs results/person4_100k_seed_0 results/person4_100k_seed_1 results/person4_100k_seed_2 \
  --run-dir results/person4_100k_aggregate
```

Aggregation checks dataset, environment spec, checkpoint preprocessing, algorithm
settings and evaluation seeds. It rejects repeated training seeds. Missing matching
steps are marked as partial comparisons. Small reviewed reports can be committed
under `docs/evidence/person-4/`; full checkpoints/logs/videos remain in ignored artifact
directories. Large artifacts should be uploaded to team storage or a release and
their URLs/checksums added before claiming the remote-publication acceptance gate.

The measured 100k-update artifacts are available in the
[Person 4 release](https://github.com/MedvAx-AI/iql-halfcheetah/releases/tag/person4-evaluation-100k).
See the [report](evidence/person-4/README.md), [video links](../videos/README.md)
and [publication manifest](evidence/person-4/publication.json). The fixed-seed demos
fall or stall; these runs do not establish robust locomotion.

MP4 recording draws the MuJoCo floor plane as infinite, matching its infinite
collision geometry. This avoids an apparent edge at the default ±40 m visual
rectangle. Only drawing extents change; observations, actions, rewards and episode
boundaries are preserved. Falls remain in the full episode video.

## Verification and limits

`tests/test_evaluate.py` checks saved preprocessing, raw reward sums, both ending
flags, seeded environment/policy repeatability, per-episode random streams, invalid
actions, environment cleanup, actual MP4 decoding, failed-video cleanup, checkpoint
loading/RNG isolation, CSV reruns, manifests and separate statistics. Required project
lint/format, tests, environment and notebook checks also apply.

A short run can verify the pipeline while performing poorly. Numerical losses are
not returns. Record the actual updates, hardware, wall time and all failed runs with
measured results; never infer improvement from passing tests. Clean Colab acceptance
and final slide integration remain the lead/Person 5's integration work.
# Platform scope

Seeded results reproduce on the same platform/locked stack. Do not assume exact
returns across macOS, Linux and Windows; label these separately. The notebook now
replays all three published final checkpoints twice and compares all 60 episode
records, then displays the current-platform aggregate. See
[reproducibility investigation](REPRODUCIBILITY.md) and
[Colab execution](COLAB_ACCEPTANCE.md).

