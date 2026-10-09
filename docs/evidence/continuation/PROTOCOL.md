# Continuation experiment — 9 October 2026

Question: does continuing the published 100k IQL checkpoints improve the observed fall/stall behavior, without changing the task, rewards, observation normalization, evaluation seeds or algorithm?

Three independent training seeds (0, 1, 2) resume their published checkpoint and saved dataset sampler/optimizer state. Candidate checkpoints: 150k, 200k, 300k and 500k. Training and paired baseline evaluation run on Windows CPU, Python 3.11.15, PyTorch 2.7.1+cpu, NumPy 2.2.6 and MuJoCo 3.2.3. Original training was on macOS; these are explicitly continuation experiments, not an exact reconstruction of the original training trajectory.

Each baseline and candidate uses the same evaluation seeds 10000–10009 and unmodified 1,000-step environment. Raw CSVs, JSON summaries, checkpoint hashes and descriptive per-episode diagnostics are retained. No favorable episode replaces the predeclared demonstration seed 10000. Selected candidates must also be checked on previously unused evaluation seeds 20000–20009 before a generalization claim.

An improvement requires higher paired mean raw return and better late-episode forward movement than that training seed's 100k baseline. A high total return alone is insufficient to call the fall/stall problem resolved. Inspect the full predetermined demo for sustained movement and falls; report low torso height and stationary tails as descriptive diagnostics, not standardized HalfCheetah success scores. Keep unsuccessful outcomes and the existing published checkpoint/demo if improvement is not demonstrated.

The plot change is independent: single-checkpoint figures show separate seed categories, every raw episode, and mean ± population episode SD. Multiple-checkpoint figures keep the quantitative update axis. Neither episode SD nor across-seed SD is a confidence interval.

The first exploratory local run used a stale MuJoCo 3.3.7 environment. Its exploratory_mujoco_3_3_7.json at the experiment root is excluded from comparison. Only continuation_seed_*/baseline_diagnostics.json, explicitly recording MuJoCo 3.2.3, supplies paired baselines.
