# Integration design

Scope: Person 1 supplies packaging, configuration, contracts, CI, workflow and task handoffs.
Persons 2–5 supply dataset/environment, IQL/training, evaluation and learning materials.

Use one installable Python 3.11 package, `iql_project`, under `src/`. Keep each
owner's implementation in separate modules and branches. No training, data download
or network requests on import. Unimplemented operations raise `NotImplementedError`
with their owner; a scaffold check must not claim algorithm readiness.

Contracts use float32 NumPy batches: observations/next observations `(B,17)`,
actions `(B,6)`, rewards `(B,1)`, terminations/truncations bool `(B,1)`.
The IQL implementation converts batches to tensors on its selected device.
Keep terminations and truncations separate. Time-limit truncation ends evaluation
episodes but does not mask the bootstrap; target uses `1 - terminated`.

Minari dataset: `mujoco/halfcheetah/medium-v0`. Evaluation must recover its
environment specification and reuse training observation normalization. The
artifact contract carries normalization, config, dataset identity and versions.

Default config contains proposed hyperparameters, not validated performance claims.
Person 3 owns IQL networks, the offline update, checkpoints and `train`
(`docs/IQL.md`). Person 4 owns evaluation/recording. Person 5 owns the notebook text,
manual example and slides. Person 1 later integrates their PRs and performs clean
Colab Run all after those deliverables exist.

Verification now: isolated dependency install, dependency consistency, packaging,
lint, contract tests, notebook validity, real HalfCheetah reset/step. No training,
dataset download, evaluation results or demo is claimed by this scaffold.
