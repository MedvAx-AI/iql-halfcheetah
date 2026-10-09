# Scaffold verification and handoff

Checked on **2026-10-04**. This report concerns Person 1's scaffold; it does not
claim an implemented algorithm, trained checkpoint, results or final Colab.

## Local checks

Reference local runtime: Windows x64, Python 3.11.15, isolated `.venv` created
from `uv.lock` using uv 0.8.22 with dev/notebook extras and CPU PyTorch 2.7.1.

- Dependency consistency: 155 installed packages compatible (`uv pip check`).
- Contract suite: 14 tests passed, including invalid config and explicit unfinished
  train/evaluate exits with no artifacts. First run failed before package creation.
- Ruff lint and format checks: passed.
- Real Gymnasium HalfCheetah-v5: reset and ten steps, finite observations/rewards.
- Notebook: nbformat schema valid; code outputs empty and execution counts cleared.

The pinned numerical environment works locally. No dataset is downloaded and no
training or rendering is performed. Linux checks run in GitHub Actions after publication.
Wheel and source distribution built successfully in isolated build environments.
Minari CLI help also executed successfully. The pip requirements wrapper was checked
with an installation dry run against the resolved environment; no changes required.

## Remote checks and review

- Initial baseline commit: `f55017a04c1b36f6fec2ab0ad387c59f00291891`.
- [Linux CI run](https://github.com/MedvAx-AI/iql-halfcheetah/actions/runs/37197602855):
  completed successfully, check name `Scaffold checks`. Includes frozen install,
  dependency/export consistency, lint, all tests, real environment smoke, notebook
  schema and wheel/sdist build on Python 3.11.
- Independent read-only scaffold review: no Critical, Important or Minor findings.
  Algorithm/data/evaluation/finished notebook correctly remain outside M0 scope.
- Four role issues and four roadmap milestones created; see [team handoff](TEAM_HANDOFF.md).

## Git workflow configuration

Feature branches start from the finalized M0 `main` commit. Squash merge is enabled;
merge commits/rebase merge and automatic branch deletion are disabled.
`main` protection requires the `Scaffold checks` status, up-to-date branches, one
approving review, code-owner review, resolved conversations and linear history.
Stale approvals are dismissed; force pushes and branch deletion are disallowed.
Administrator enforcement is disabled so the lead retains an explicit maintenance
bypass; contributors follow the protected PR path. See the live branch settings
for any future changes to enforcement.

## Final team checks required at scaffold delivery (historical)

Dataset metadata and episode-boundary validation (Person 2); IQL numerical correctness,
real-data updates, checkpoint/resume and GPU profile if used (Person 3); actual
seeded returns and headless video (Person 4); completed narrative/manual example/slides
(Person 5); integrated fresh Colab Run all and final release (Person 1).

Participant usernames were not provided. Role issues are intentionally unassigned;
the lead assigns them and grants access when those usernames are available.

## Final integration acceptance — 9 October 2026

The subsequent role implementations are merged. A fresh Google Colab CPU
**Run all** at `e710dcb33621861c9c1d1e14735a3fd7a0beb8dd` completed all 11 code
cells, 1,000 real-data training updates, exact repeatability of 60 checkpoint
evaluation records, new video decoding and all 92 tests. The default Python 3.13
Colab host automatically used the frozen Python 3.11 project kernel.

Full evidence and limitations: [Colab acceptance](COLAB_ACCEPTANCE.md).
The earlier scaffold checklist and skipped Windows smoke are historical records,
not the basis for this acceptance. Cross-platform policy returns are separately
labeled; robust running and the default 500k schedule are not claimed.
