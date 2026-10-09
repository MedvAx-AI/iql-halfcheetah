# Continuation experiment — 9 October 2026

Question: does continuing the published 100k IQL checkpoints improve the observed fall/stall behavior, without changing the task, rewards, observation normalization, evaluation seeds or algorithm?

Three independent training seeds (0, 1, 2) resume their published checkpoint and saved dataset sampler/optimizer state. Candidate checkpoints: 150k, 200k, 300k and 500k. Training and paired baseline evaluation run on Windows CPU, Python 3.11.15, PyTorch 2.7.1+cpu, NumPy 2.2.6 and MuJoCo 3.2.3. Original training was on macOS; these are explicitly continuation experiments, not an exact reconstruction of the original training trajectory.

Each baseline and candidate uses the same evaluation seeds 10000–10009 and unmodified 1,000-step environment. Raw CSVs, JSON summaries, checkpoint hashes and descriptive per-episode diagnostics are retained. No favorable episode replaces the predeclared demonstration seed 10000. Selected candidates must also be checked on previously unused evaluation seeds 20000–20009 before a generalization claim.

An improvement requires higher paired mean raw return and better late-episode forward movement than that training seed's 100k baseline. A high total return alone is insufficient to call the fall/stall problem resolved. Inspect the full predetermined demo for sustained movement and falls; report low torso height and stationary tails as descriptive diagnostics, not standardized HalfCheetah success scores. Keep unsuccessful outcomes and the existing published checkpoint/demo if improvement is not demonstrated.

The plot change is independent: single-checkpoint figures show separate seed categories, every raw episode, and mean ± population episode SD. Multiple-checkpoint figures keep the quantitative update axis. Neither episode SD nor across-seed SD is a confidence interval.

The first exploratory local run used a stale MuJoCo 3.3.7 environment. Its exploratory_mujoco_3_3_7.json at the experiment root is excluded from comparison. Only continuation_seed_*/baseline_diagnostics.json, explicitly recording MuJoCo 3.2.3, supplies paired baselines.

## Reproduce the continuation

The [recorded runner](run_continuation.py) is the actual experimental script,
with formatting changes only. Run from the repository root in its locked
environment. First retrieve and verify the original checkpoints with
`scripts/replay_published.py` after downloading the dataset:

```bash
uv sync --frozen --extra dev --extra notebook
uv run --frozen python -c "import minari; minari.load_dataset('mujoco/halfcheetah/medium-v0', download=True)"
uv run --frozen python scripts/replay_published.py --output-dir artifacts/original-checkpoints
uv run --frozen python docs/evidence/continuation/run_continuation.py \
  --checkpoint artifacts/original-checkpoints/checkpoints/person4_100k_seed_0/step_100000.pt \
  --run-dir artifacts/continuation_seed_0 --steps 150000 200000 300000 500000
```

Repeat for training seeds 1 and 2 with their corresponding original files and
distinct output directories. The runner resumes the returned checkpoint after
each measured stage; no algorithm or dataset settings are overridden. It saves
every 50k checkpoint, logs every update and evaluates the specified stages.
Use a fresh run directory. Exact training trajectories and returns can differ
on another CPU/OS despite matching dependency versions.

Before treating a new run as using the identical dataset, confirm the dataset
report's `main_data.hdf5` SHA-256 is
`db46045497395bc621f2b409089e495f0e51a1182e6257f2f319f7d6654f66c5`
with `uv run --frozen python scripts/check_dataset.py --output artifacts/dataset-report.json`.

## Reproduce the recorded all-stage evaluation and curves

After retrieving the original checkpoints above, the retained
[Colab audit script](audit_colab.py) evaluates all three original policies and
their four continuation stages in one runtime. The
[checkpoint inventory](checkpoint_inventory.json) verifies each downloaded
candidate before deserialization. It records all 300 actual IQL episodes on
the validation and additional sets, with descriptive velocity/posture diagnostics.

```bash
uv run --frozen python docs/evidence/continuation/audit_colab.py \
  --original-dir artifacts/original-checkpoints/checkpoints \
  --output-dir artifacts/colab-audit \
  --inventory docs/evidence/continuation/checkpoint_inventory.json
uv run --frozen python docs/evidence/continuation/summarize_colab.py
```

The second command rebuilds the committed reference figures from the retained
frontend JSONs next to the script, rather than replacing those inputs with
your new measurements. Each figure contains 150 IQL episodes: three training
seeds, five measured checkpoints and ten episodes per point. It verifies fixed
episode sets and identical paired baselines, and retains input/checkpoint hashes.
Population episode SD and sample SD across three run means remain separate.

The initial Colab session disconnected while continuation jobs finished locally.
All five stages were then measured in one reconnected session for the final
reference curves. Earlier observed audits remain in `prior-session/` and the
experimental release; they are not pooled with the final session's results.
The sets 20000–20009 were inspected repeatedly during selection and are described
as an additional evaluation set, not an untouched final test set.
