# Person 2 verification evidence

The [dataset report](dataset_report.json) was produced from clean implementation
commit `eb3fc2f9c307c20260a9bea0974a6e5b9c631030` on Windows / Python 3.11.16.
The following evidence-only commit publishes this report; no implementation or
configuration changes occur between the measured source and the PR head.

Config: `configs/halfcheetah.toml`; dataset: `mujoco/halfcheetah/medium-v0`; seed: `0`.
The report includes config, versions, recovered EnvSpec, preprocessing, data-file
hashes and hashes of the exact implementation/config files that were checked.

Report SHA-256:
`32ea8fc6ae82f3012e369bf91408842b8a46dc2c32cee4f339233d4b715c840e`.

Exact local validation commands (run from the repository root):

```powershell
.\.venv\Scripts\python.exe scripts/check_dataset.py --config configs/halfcheetah.toml --datasets-path data/minari
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe scripts/check_notebook.py
.\.venv\Scripts\python.exe -m iql_project check --config configs/halfcheetah.toml
uv pip check --python .venv/Scripts/python.exe
uv export --frozen --all-extras --no-emit-project --no-hashes --no-header --no-cache -o results/person-2/requirements.verify.txt
.\.venv\Scripts\python.exe -c "from pathlib import Path; assert Path('requirements.lock.txt').read_text() == Path('results/person-2/requirements.verify.txt').read_text()"
.\.venv\Scripts\python.exe -m build --installer uv
git diff --check
```

Results: 49 tests passed; lint and formatting passed; all 1,000,000 transitions
validated; seeded batches equal; recovered environment reset plus 10 finite steps
passed; notebook schema/output checks passed; installed dependencies compatible;
exported requirements match; wheel and source distribution built successfully.

The initial pip-based isolated build encountered a machine certificate-chain
error. Retrying with `--installer uv` succeeded with the same pinned build
dependencies. The Linux GitHub workflow remains unchanged and reports its own
status on the PR; local success does not stand in for CI success.

This original report used MuJoCo 3.3.7. The current pin is now 3.2.3, matching
collection metadata; see [simulator compatibility](../../SIMULATOR_COMPATIBILITY.md)
for subsequent verification. The original report and its checksum are preserved.
See [the handoff](../../ENVIRONMENT_DATASET.md) for the undeclared dataset license. Only the small JSON
report is published; the dataset, cache, build files and other generated artifacts
remain ignored. Shared API signatures are unchanged; the Person 2 notebook
sections use the public package API and contain no executed outputs.
