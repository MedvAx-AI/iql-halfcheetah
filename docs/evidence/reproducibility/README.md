# Simulator-only platform probe

Recorded 2026-10-08 with Gymnasium 1.2.2, MuJoCo 3.2.3 and NumPy 2.2.6. No IQL
network is used. `simulator_platform_probe.py` resets HalfCheetah-v5 at seed 10000
and applies the same pre-generated 1,000 float32 uniform random actions.

Windows x86_64 and Google Colab Linux x86_64 records are in `windows.json` and
`colab.json`. Initial qpos/qvel, action-tape and XML hashes match exactly. The
compiled model parameter hashes differ. Maximum absolute differences between the
recorded qpos/qvel state vectors are:

| Environment step | Maximum absolute difference |
| --- | ---: |
| 1 | 1.1102230246251565e-16 |
| 10 | 2.042810365310288e-14 |
| 100 | 0.00033415021871474604 |
| 1000 | 5.059187583970611 |

These are simulator state-vector differences, not policy returns. They establish a
platform-dependent model/simulation path even with identical reset/control bytes.
The macOS experiment was not available for this simulator-only probe; no complete
compiler-level attribution of its policy return difference is claimed.

Reproduce with the frozen environment:

```bash
uv run --frozen python docs/evidence/reproducibility/simulator_platform_probe.py new-probe.json
```
