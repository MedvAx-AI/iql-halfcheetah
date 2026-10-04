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
