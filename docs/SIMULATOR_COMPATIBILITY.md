# Simulator version decision

The `mujoco/halfcheetah/medium-v0` dataset metadata records `mujoco==3.2.3` and
`gymnasium>=1.0.0`. The project's original 3.3.7 engine pin triggered Minari's
requirement warning when recovering HalfCheetah-v5.

Use **MuJoCo 3.2.3** with the existing Gymnasium 1.2.2 / Minari 0.5.3 pins.
`pyproject.toml`, `uv.lock` and the pip requirements export agree on this version.
Update an existing environment with:

```bash
uv sync --frozen --extra dev --extra notebook
```

The real Minari HDF5 roundtrip test now records the same engine requirement and
treats a simulator version warning as a failure. It failed under 3.3.7 before the
dependency correction; the CI test prevents accidental reintroduction.

Person 2's original verification report and checksum remain unchanged as historical
evidence. Subsequent verification uses the new locked environment and the exact
official HDF5 data hash. Matching the engine requirement removes the known version
mismatch; it does not establish bitwise equivalence with every original collector
dependency. Final trained-policy results and video remain future team deliverables.

## Verified on 2026-10-04

- Python 3.11.15 on Windows x64, locked MuJoCo 3.2.3 environment.
- All 49 tests passed, including real Minari HDF5 recovery with the collection
  engine requirement enforced as an error on mismatch.
- Full official dataset: 1,000,000 transitions / 1,000 episodes validated, seeded
  sampling matched, recovered environment reset and ten finite steps passed with
  `-W error::UserWarning`. The previous 3.3.7 environment failed that same check.
- Preprocessing, raw statistics and HDF5 checksum exactly match Person 2's report.
- Dependency consistency, lint/format, notebook validation, exported requirements,
  and isolated wheel/source builds passed.

The new [dataset report](evidence/simulator-3.2.3/dataset_report.json) records clean
implementation commit `1cf7634`, package versions and source/data hashes. Its SHA-256
for the Git-stored LF bytes is
`8c633ad873aa91d0462bed837e0bd9c97da1acffda9b74b5f1456efa471bfd3a`.

Reproduce the full check after an explicit dataset download:

```bash
uv sync --frozen --extra dev --extra notebook
uv run --frozen python -W error::UserWarning scripts/check_dataset.py --datasets-path data/minari
```
