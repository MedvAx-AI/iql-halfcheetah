# Team handoff

Repository: [MedvAx-AI/iql-halfcheetah](https://github.com/MedvAx-AI/iql-halfcheetah).
Lead and code owner: @MedvAx-AI. Baseline: `main`.

| Role | Created branch | Created issue |
|---|---|---|
| Person 2 — environment/data | [feature/environment-dataset](https://github.com/MedvAx-AI/iql-halfcheetah/tree/feature/environment-dataset) | [#1 — environment and Minari dataset](https://github.com/MedvAx-AI/iql-halfcheetah/issues/1) |
| Person 3 — IQL/training | [feature/iql](https://github.com/MedvAx-AI/iql-halfcheetah/tree/feature/iql) | [#2 — IQL, training and checkpoint/resume](https://github.com/MedvAx-AI/iql-halfcheetah/issues/2) |
| Person 4 — evaluation/results | [feature/evaluation](https://github.com/MedvAx-AI/iql-halfcheetah/tree/feature/evaluation) | [#3 — evaluation, plots and video](https://github.com/MedvAx-AI/iql-halfcheetah/issues/3) |
| Person 5 — notebook/slides | [feature/notebook-presentation](https://github.com/MedvAx-AI/iql-halfcheetah/tree/feature/notebook-presentation) | [#4 — notebook, manual example and slides](https://github.com/MedvAx-AI/iql-halfcheetah/issues/4) |

Each issue includes measured acceptance checks, dependencies, branch name and
review evidence. Role labels are set; assignees await actual participant usernames.
The public repository can be cloned/forked immediately; direct role-branch pushes
require collaborator access granted by the lead.

## Roadmap

- [M1 — proposal/data contract](https://github.com/MedvAx-AI/iql-halfcheetah/milestone/1)
- [M2 — minimal integrated pipeline](https://github.com/MedvAx-AI/iql-halfcheetah/milestone/2)
- [M3 — reproducible experiments](https://github.com/MedvAx-AI/iql-halfcheetah/milestone/3)
- [M4 — final notebook/defense](https://github.com/MedvAx-AI/iql-halfcheetah/milestone/4)

No dates are invented. Participants can start independently with small fixtures,
evaluation protocols and narrative; integrated acceptance follows the milestone gates.

## Lead's integration gates at M0 (historical)

1. Review Person 2's real dataset/environment PR and freeze preprocessing metadata.
2. Merge tested IQL/training; verify one real-data update and checkpoint parity.
3. Integrate evaluation and verify results/video provenance across seeds.
4. Verify Person 5's completed notebook in a clean Colab, check artifact links and
   record commit/runtime. Tag the final course release after all gates pass.

The M0 team-lead setup is complete; these subsequent gates depend on the team
implementations. No training or final-demo work was substituted for Persons 2–5.

## Final delivery — 9 October 2026

All four integration gates above are complete. Persons 2–5 supplied their role
implementations; PR #13 supplied automatic Colab setup and platform-scoped replay
verification. The actual merged notebook passed a fresh default Colab **Run all**:
11 code cells, 92 tests, real-data training, checkpoint replay, plot and video.

The tested commit, raw results and scope are in [COLAB_ACCEPTANCE.md](COLAB_ACCEPTANCE.md).
The final course snapshot is published under the `v0.1.0-course` release tag.
Course delivery retains measured limitations; it does not claim robust locomotion,
500k-update experiments or completion of the team's oral defense.
