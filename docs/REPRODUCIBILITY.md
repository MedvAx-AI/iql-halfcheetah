# Reproducibility scope

Seeded replay is checked **on the same platform and locked environment**. We do
not promise identical episode returns across macOS arm64, Linux x86_64 and Windows
x86_64. Do not combine their results as if they were independent training seeds
from one experiment.

## Measured difference and investigation

The original Person 4 Apple M3 Pro / macOS CPU experiment returned **2846.41 ±
786.80** across three final 100k-update run means. A replay of the same checkpoint
bytes on Google Colab Linux x86_64 returned **2187.95 ± 480.18**. The Windows x86_64
replay returned **2962.99 ± 502.81**. These spreads are sample standard deviations
across run means, not episode standard deviations or confidence intervals.

The dataset SHA-256, checkpoint SHA-256, configuration, evaluation seeds 10000–10009,
episode horizon, saved preprocessing and numerical dependency versions were checked.
The random baseline also differs, although it does not use a neural network.

A separate simulator-only probe on Windows and Colab used the same HalfCheetah XML,
bit-identical initial qpos/qvel and identical 1,000 float32 actions generated with
seed 10000. With MuJoCo 3.2.3 and NumPy 2.2.6, their physics states already differ at
floating-point precision after the first environment step and diverge substantially
by step 1,000. Compiled model parameter hashes differ too. This establishes a
platform difference in the simulator/model path independently of IQL. It does not
identify one compiler instruction as the source, nor isolate every contribution to
the macOS-versus-Colab policy-return difference. Probe data and the executable probe
are retained under `docs/evidence/reproducibility/`.

This behavior agrees with [MuJoCo's documented reproducibility scope](https://mujoco.readthedocs.io/en/3.7.0/computation/index.html#reproducibility):
small numerical differences can grow during contact simulation, and exact reproduction
has an architecture/version scope. Matching version numbers and RNG seeds is useful,
but does not make different platform binaries interchangeable.

## Reproduction and acceptance

Run from a locked environment with the Minari dataset already downloaded:

```bash
uv run --frozen python scripts/replay_published.py --output-dir results/replay-new --video
```

The script verifies the published archive and all three final checkpoint hashes,
performs two independent evaluations for each training seed, compares all **60
individual IQL/random episode records**, and records `repeatability.json`, raw CSVs,
manifests, plots and an aggregate. Different run names are ignored; episode values,
lengths, evaluation seeds, checkpoint steps and policy labels must agree exactly.
Use a fresh output directory when recording a new video.

The notebook runs this check automatically, displays **its current platform's**
aggregate, and keeps the original macOS results separately labeled in the narrative.
Its fresh 1,000-update training smoke is separate from the three original 100k runs.
Every acceptance report records actual source commit, project Python and platform.

For stronger future comparisons, use the same OS/architecture/runtime for training
and evaluation, record every dependency/binary and increase evaluation coverage.
No new performance threshold or claim of robust locomotion follows from this fix.
