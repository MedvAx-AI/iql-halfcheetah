# Final presentation slides — Person 5

Numbers match `docs/evidence/person-4/summary.json` and `docs/evidence/person-4/README.md`.

---

## Slide 1 — Problem

Offline RL on HalfCheetah: learn from a **fixed** Minari dataset only, then control HalfCheetah-v5 online. Challenge: avoid overestimating actions missing from the data.

---

## Slide 2 — Environment

**HalfCheetah-v5** · obs `(17,)` · action `(6,)` in `[-1, 1]` · forward reward − control cost · 1,000-step horizon · EnvSpec recovered from Minari (MuJoCo 3.2.3).

---

## Slide 3 — Dataset

`mujoco/halfcheetah/medium-v0` · 1,000 episodes / 1,000,000 transitions · float32 training contract · truncation keeps Q bootstrap; termination would mask it (none in this dataset).

---

## Slide 4 — IQL (three losses)

| Loss | Role |
|---|---|
| Expectile V | Upper expectile of in-dataset Q (`τ=0.7`) |
| Masked Q | `y = r + γ (1−terminated) V(s')` |
| Capped AWR policy | `w = min(exp(β A), 100)`, **β = inverse temperature = 3** |

---

## Slide 5 — Manual numerical example

`Q=8`, `V=5`, `V'=6`, `r=1`, `γ=0.99`, `τ=0.7`, `β=3`:

- Expectile weight on `u=3`: `0.7` → V term `6.3`
- Q target (not terminated): `6.94` (terminated → `1.0`)
- Policy weight: `exp(9) ≈ 8103` → **capped to 100**

---

## Slide 6 — Experiment protocol

- Config base: `configs/halfcheetah.toml`
- **100k** offline updates (not 500k), checkpoints every 25k
- Training seeds `0,1,2` · eval seeds `10000–10009`
- Hardware: Apple M3 Pro CPU · ≈322–330 s / seed
- Artifacts: `docs/evidence/person-4/` + release `person4-evaluation-100k`

---

## Slide 7 — Measured results (raw returns)

| Policy | Mean return | Spread |
|---|---:|---:|
| Random baseline | **−287.40** | ±56.68 (episode) |
| IQL @ 100k (across 3 seeds) | **2846.41** | ±786.80 (sample std of run means) |
| Seed 0 / 1 / 2 finals | 1964 / 3475 / 3099 | large episode std |

Learning is **not monotonic** (seed 0 was higher at 50k). Plot: `docs/evidence/person-4/evaluation_returns.png`.

The table is the **original macOS / Apple M3 Pro experiment**. Replaying the same
checkpoint bytes on Colab Linux x86_64 produced **2187.95 ± 480.18** and random
baseline **−302.78**. Run all computes fresh platform-specific results and verifies
every episode against a repeat on that platform. Cross-platform exact equality is
outside MuJoCo's guarantee; simulator-only probes demonstrate divergence from
identical initial states/actions. See `docs/REPRODUCIBILITY.md` and the Colab evidence.

---

## Slide 8 — Demo, limitations, references

**Demo:** [seed 0](https://github.com/MedvAx-AI/iql-halfcheetah/releases/download/person4-evaluation-100k/person4_100k_seed_0_visible_floor_demo_seed_10000.mp4) · [seed 1](https://github.com/MedvAx-AI/iql-halfcheetah/releases/download/person4-evaluation-100k/person4_100k_seed_1_visible_floor_demo_seed_10000.mp4) · [seed 2](https://github.com/MedvAx-AI/iql-halfcheetah/releases/download/person4-evaluation-100k/person4_100k_seed_2_visible_floor_demo_seed_10000.mp4)

**Honest takeaway:** returns beat random, but all three fixed demos **fall/stall** — not robust running.

**Reproducible demo:** the Colab notebook performs real data loading, 1,000 fresh
training updates, checkpoint retrieval/replay, MP4 recording and the full tests
without manual setup. This training smoke is separate from the 100k experiment.

**References:** Kostrikov et al. ICLR 2022; Minari medium-v0; Gymnasium HalfCheetah; `docs/INTERFACES.md`, `docs/EVALUATION.md`.
