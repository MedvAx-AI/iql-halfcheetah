# Person 2 — environment and offline data

Implements [issue #1](https://github.com/MedvAx-AI/iql-halfcheetah/issues/1) and
the Person 2 section of [the project plan](../IQL_5_Person_Project_Plan.md).
Shared configuration and contracts are unchanged. IQL, training, evaluation,
videos and presentation work remain with their owners.

## Environment

HalfCheetah-v5 controls a planar robot. Its 17 observations are eight generalized
positions (excluding absolute horizontal position) and nine velocities. Its six
actions apply torques to the back/front thigh, shin and foot, each in `[-1, 1]`.
The objective is fast forward motion with low control effort. With this dataset's
recorded default kwargs, reward is horizontal velocity minus
`0.1 * sum(action**2)`. Each step advances five simulation frames (`dt = 0.05 s`).
The horizon is 1,000 steps; HalfCheetah does not terminate on falling.
The Gymnasium time limit produces truncation instead.

These continuous states/actions are suitable for IQL's value/Q networks and
advantage-weighted policy imitation from fixed transitions. Environment interaction
is needed for subsequent evaluation, not for collecting online training data.

`make_evaluation_env(config, render_mode=None)` loads the local dataset and calls
Minari's `recover_environment` with the collection EnvSpec. It preserves recorded
reward weights, observation settings and horizon, validates recovered spaces/bounds,
and accepts `render_mode="rgb_array"` for Person 4. It never downloads data or
substitutes a newly constructed environment with default settings.

References: [Gymnasium HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)
and [Minari dataset card](https://minari.farama.org/datasets/mujoco/halfcheetah/medium-v0/).
Installed Gymnasium 1.2.2 source was also checked for the reward and observation formulas.

## Loading and handoff to Person 3

Install the existing locked dependencies as described in the root README. Download
only by an explicit command or `download=True` call. To keep data inside the ignored
workspace cache, use these PowerShell commands:

```powershell
$env:MINARI_DATASETS_PATH = Join-Path (Get-Location) 'data/minari'
uv run --frozen python scripts/check_dataset.py --download --datasets-path data/minari
```

Subsequent checks use no network:

```powershell
uv run --frozen python scripts/check_dataset.py --datasets-path data/minari
```

The script's `--datasets-path` selects the cache for that process only. Set
`MINARI_DATASETS_PATH` in the caller's environment for the notebook/training too,
or use the default Minari user cache consistently. If data is absent, the loader
raises `FileNotFoundError` with an explicit download instruction. Imports have no
downloads, learner initialization or rendering side effects.

```python
import numpy as np
from iql_project.config import load_config
from iql_project.dataset import load_offline_dataset

config = load_config("configs/halfcheetah.toml")
dataset = load_offline_dataset(config)  # download=False
rng = np.random.default_rng(config.seed)
batch = dataset.sample(config.batch_size, rng)
dataset.save_metadata("results/person-2/preprocessing.json")
```

`batch` is the shared `TransitionBatch` dataclass, not the plan's illustrative
dictionary. Observations/next observations have shape `(B, 17)`, actions `(B, 6)`,
rewards `(B, 1)`, all float32. `terminated` and `truncated` are separate bool
arrays `(B, 1)`. Sampling is uniform with replacement; reuse the caller's RNG
across updates and save its state for resuming. A batch contains writable copies,
so modifying one cannot corrupt the dataset. The learner owns PyTorch conversion.

Episodes contain T actions/rewards/flags and T+1 observations. The loader pairs
`observations[:-1]` with `observations[1:]` separately inside each episode.
It rejects malformed shapes, non-bool flags, early ending flags, invalid values,
float32 overflow, out-of-bound actions/observations and incorrect metadata counts.
An unflagged final transition is retained as a partial episode, without inventing
an ending flag. A truncated last transition retains its own recorded final state.
Use `1 - batch.terminated` for bootstrapping; truncation alone does not mask it.

## Preprocessing and saved statistics

Mean and population standard deviation are fitted on the offline *current*
observations (`[:-1]` in each episode) after conversion to float32, with float64
accumulation. Terminal next states are not additional samples for fitting.
The stored float32 denominator is `max(population_std, 1e-3)` per dimension.
Current and next observations use the same stored values. Rewards are unchanged
(`reward_scale=1`, `reward_shift=0`); actions retain environment units and bounds.

Person 3 should persist `dataset.info` in checkpoints. Person 4 preprocesses each
raw evaluation observation with those saved values, without refitting:

```python
observation = np.asarray(raw_observation, dtype=np.float32)
preprocessed = (observation - info.observation_mean) / info.observation_std
action = policy.act(preprocessed, deterministic=True)
```

`dataset.metadata` returns a copy with ID, dimensions, episode/transition counts,
source metadata, collection requirements, EnvSpec, raw reward/return/length
statistics and exact normalization values. `save_metadata(path)` writes that
record as JSON. Summary statistics use untransformed float32 reward values and
population std (`ddof=0`); these are dataset statistics, not trained-agent scores.
The in-memory transitions use approximately 158 MiB, plus loading/statistics overhead.

## Measured full-dataset verification

Verified locally on 2026-10-04 with Python 3.11.16, NumPy 2.2.6, Minari 0.5.3,
Gymnasium 1.2.2, MuJoCo 3.3.7 and CPU PyTorch 2.7.1. Config:
`configs/halfcheetah.toml`; seed: `0`. The committed evidence report records the
tested implementation commit, clean-working-tree flag and code/config SHA-256
values. A subsequent evidence-only commit adds that report for review.

| Statistic | Measured value |
|---|---:|
| Episodes | 1,000 |
| Transitions | 1,000,000 |
| Observation / action dimension | 17 / 6 |
| Reward mean / std | 12.08921036 / 4.69655068 |
| Reward min / max | -3.15781450 / 17.46031952 |
| Episode return mean / std | 12,089.21035772 / 3,008.94886216 |
| Episode return min / max | 234.62614191 / 14,238.90506859 |
| Episode length (all episodes) | 1,000 |
| Terminated / truncated transitions | 0 / 1,000 |
| Partial episodes | 0 |

The recovered environment reset and completed 10 finite random-action steps.
All transitions were validated; two batches drawn with identical RNG seeds were equal.
The generated report is `results/person-2/dataset_report.json` (ignored by Git).
The small reviewed copy is [docs/evidence/person-2/dataset_report.json](evidence/person-2/dataset_report.json).
The script prints its SHA-256 and includes the data-file checksums; the PR records
the published report's checksum and exact validation commands. Rerunning produces
a new report/checksum because elapsed time is recorded.

The downloaded HDF5 matches the official mirror's LFS SHA-256:
`db46045497395bc621f2b409089e495f0e51a1182e6257f2f319f7d6654f66c5`
(210,168,080 bytes). Source:
[official Farama mirror](https://huggingface.co/datasets/farama-minari/mujoco/tree/main/halfcheetah/medium-v0/data).
Minari's Python downloader failed DNS resolution on this machine, so the two
official files were downloaded with PowerShell into the same Minari cache layout.
The acceptance script then ran without `--download`. No download workaround is
embedded in the loader, and the full dataset is excluded from Git.

Source author: Kallinteris Andreas; collection algorithm: SB3/TQC; dataset Minari
version: 0.5.2. **Dataset license is not declared in the downloaded metadata or
mirror's dataset card**; `source.license` is recorded as `null`. A collection-code
license must not be treated as a dataset license.

**Simulator version decision:** metadata requires `mujoco==3.2.3` and `gymnasium>=1.0.0`.
The team's dependency pin is now MuJoCo **3.2.3**, matching that requirement.
Run `uv sync --frozen --extra dev --extra notebook` to update an existing environment.
Person 2's original report above used the earlier 3.3.7 runtime and its version
warning; that report is historical evidence and has not been rewritten.
See [simulator compatibility](SIMULATOR_COMPATIBILITY.md) for the version decision
and subsequent verification. Matching the engine requirement does not itself prove
bitwise trajectory parity with the original collector's full software stack.
Rendering/video and learner updates remain their respective owners' acceptance work.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe scripts/check_dataset.py --datasets-path data/minari
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe scripts/check_notebook.py
```

49 tests pass, including the original scaffold tests and 35 Person 2 tests.
Small fixtures check episode boundaries, last-transition flags, normalization
roundtrip, seeded sampling, shape/dtype validation, invalid-value rejection,
explicit download opt-in and nondefault environment kwargs/horizon. A real local
Minari HDF5 fixture verifies storage loading and environment recovery without network.
Lint, formatting, scaffold CLI, environment smoke and notebook validation pass.
Changes are submitted through `feature/environment-dataset` to `main` for lead
review. Merging remains the lead's decision.
