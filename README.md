# Implicit Q-Learning on HalfCheetah-v5

Five-person offline reinforcement learning course project led by **MedvAx-AI**.

**Status: Person 2 environment/data implemented.** The install, configuration,
interfaces, validated Minari loader and environment recovery are ready.
IQL, training, evaluation and video recording deliberately raise `NotImplementedError`.
No trained agent, measured agent return or completed course demo is supplied at this stage.

## Start here

- [Project plan, role deliverables and roadmap](IQL_5_Person_Project_Plan.md)
- [Person 2 environment, preprocessing and measured dataset statistics](docs/ENVIRONMENT_DATASET.md)
- [Shared interfaces and artifact formats](docs/INTERFACES.md)
- [Contribution / Git workflow](CONTRIBUTING.md)
- [Team tasks](https://github.com/MedvAx-AI/iql-halfcheetah/issues)
- [Created branches, issues and milestones](docs/TEAM_HANDOFF.md)
- [Notebook outline](notebooks/iql_halfcheetah.ipynb)
- [Verification evidence](docs/VERIFICATION.md)

## Chosen stack

Python **3.11**; PyTorch **2.7.1**; Gymnasium **1.2.2**; MuJoCo **3.2.3**;
Minari **0.5.3**. Exact direct pins live in `pyproject.toml`, complete dependency
resolution in `uv.lock`. The default environment uses CPU PyTorch on Linux/Windows.
GPU training needs a separate documented CUDA-compatible dependency lock and a
measured hardware budget from Person 3; changing `device` alone does not install CUDA.

Dataset: `mujoco/halfcheetah/medium-v0`, generated from HalfCheetah-v5, 1,000,000
steps / 1,000 episodes, observations `(17,)`, actions `(6,)` in `[-1, 1]`.
The source observations are float64; the shared training contract is float32.
Person 2's loader verifies metadata; the environment factory recovers the dataset's
specification. MuJoCo is pinned to the dataset's collection requirement; see
[simulator compatibility](docs/SIMULATOR_COMPATIBILITY.md). Person 2's original
report used the earlier 3.3.7 pin and is retained as historical evidence.
See the [official dataset card](https://minari.farama.org/datasets/mujoco/halfcheetah/medium-v0/)
and [Gymnasium environment documentation](https://gymnasium.farama.org/environments/mujoco/half_cheetah/).

## Install and check

Use Python 3.11 and run from the repository root. Recommended locked setup:

```bash
python -m pip install uv==0.8.22
uv sync --frozen --extra dev --extra notebook
uv run --frozen iql-project check --config configs/halfcheetah.toml
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen python scripts/check_environment.py
uv run --frozen python scripts/check_notebook.py
```

`uv sync` selects Python 3.11 from `.python-version`; it can install that runtime
if no matching interpreter is available. The check command validates config and
reports role ownership; it does not download the dataset or initialize a learner.
The environment check resets HalfCheetah and executes ten steps without rendering.

Alternative with pip (activate the created virtual environment for your shell):

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
iql-project check --config configs/halfcheetah.toml
```

`requirements.txt` supplies the CPU wheel index and includes `requirements.lock.txt`,
which is exported from the lock with dev/notebook dependencies and platform markers.
Regenerate the dependency export with:

```bash
uv export --frozen --all-extras --no-emit-project --no-hashes --no-header -o requirements.lock.txt
```

Linux is the Colab/CI reference platform; a Windows installation and numerical
environment smoke check are also recorded in the verification report. Rendering
needs a working OpenGL context; Linux headless rendering may use `MUJOCO_GL=egl`
with appropriate driver libraries. Video/rendering is Person 4's acceptance task.

## Project layout

```text
configs/halfcheetah.toml       shared starting hyperparameters
src/iql_project/
  config.py, contracts.py     Person 1: shared config and interface contracts
  dataset.py, environment.py  Person 2: validated loader, sampler and environment recovery
  networks.py, iql.py         Person 3: explicit implementation stubs
  train.py                   Person 3: training orchestration stub
  evaluate.py                Person 4: evaluation and recording stubs
  cli.py                     scaffold check; train/evaluate fail clearly
notebooks/                   Person 5: valid, unexecuted narrative outline
tests/                       scaffold contract tests; role PRs add behavioral tests
scripts/                     dependency/environment/notebook checks
  check_dataset.py           Person 2: explicit download, statistics and recovery check
data/                        local Minari cache guidance; data excluded from Git
results/, videos/, checkpoints/   artifact conventions; generated files excluded
docs/                        design, interfaces, verification, presentation outline
.github/                     CI, CODEOWNERS, issue and pull request templates
```

## Team branches

| Role | Branch | First deliverable |
|---|---|---|
| Person 2 | `feature/environment-dataset` | validated batch + recoverable environment |
| Person 3 | `feature/iql` | finite IQL update + checkpoint/resume |
| Person 4 | `feature/evaluation` | evaluation protocol + raw metrics + MP4 |
| Person 5 | `feature/notebook-presentation` | manual example + notebook + slides |

Changes enter `main` through pull requests reviewed by @MedvAx-AI.
Follow [CONTRIBUTING.md](CONTRIBUTING.md); CI is a scaffold gate, not evidence that
an agent learns. Integrate in the order data → IQL/train → evaluation → final Colab.

## Final Colab acceptance

Person 5 completes the notebook; Person 1 opens a fresh runtime at a recorded
commit and verifies Run all after the implementation and artifacts arrive.
The placeholder is not that final check. Colab's Python version can change:
record it and either use the Python 3.11 reference environment or validate and
commit a deliberate support update before claiming reproducibility.

## Sources and reuse

- [IQL paper: Kostrikov, Nair, Levine (ICLR 2022)](https://arxiv.org/abs/2110.06169)
- [Official IQL implementation](https://github.com/ikostrikov/implicit_q_learning)
- [Minari usage](https://minari.farama.org/content/basic_usage/)
- [Minari dataset card](https://minari.farama.org/datasets/mujoco/halfcheetah/medium-v0/)
- [HalfCheetah documentation](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)

No third-party implementation is vendored. If adapting code, cite the exact source
commit and comply with its license. The team must choose a repository license
before redistributing implementation or trained artifacts under new license terms.
