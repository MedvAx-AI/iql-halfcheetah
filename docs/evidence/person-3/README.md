# Person 3 verification evidence

The [1,000-update report](thousand_updates.json) was produced on the working tree
based on `ff723119d52c6fe0fc6423b06f29cc91d92caba2`, before the commit that adds
this report. SHA-256 values in the report identify `iql.py`, `networks.py`,
`train.py` and `cli.py`. Those implementation files are unchanged by the
evidence write-up.

Config: `configs/halfcheetah.toml` with `total_steps` overridden to 1,000 and
`checkpoint_interval` 1,000. Dataset: `mujoco/halfcheetah/medium-v0`. Seed: `0`.
Device: CPU. Batch size: 256. Python 3.11.15, PyTorch 2.7.1, NumPy 2.2.6.

Local commands, from the repository root:

```bash
export MINARI_DATASETS_PATH="$PWD/data/minari"
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen python scripts/check_notebook.py
uv run --frozen iql-project check --config configs/halfcheetah.toml
```

The Minari download used `load_offline_dataset(..., download=True)` inside
`train`. All 1,000 logged updates had finite `q_loss`, `v_loss`, `policy_loss`
and `advantage_mean`. `q_loss` moved from about 343 at step 1 to about 4.2 at
step 1,000. Training updates took about 5.6 seconds after the dataset was
loaded; wall clock including download and load was about 26 seconds.

`policy_loss` is `-E[w log π]`. It may be negative when the squashed Gaussian
is peaked on dataset actions. That is not a return and not divergence.

Generated `results/person3_real_1000/` and `checkpoints/person3_real_1000/` stay
out of Git. Their checksums are in the report. This run does not claim a
HalfCheetah score; evaluation is still Person 4.
