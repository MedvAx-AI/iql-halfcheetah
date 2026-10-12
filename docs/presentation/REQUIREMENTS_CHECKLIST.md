# Proposal requirements coverage

Source: the two-page course handout `RL_Project_Proposal_Presentation.pdf` supplied by the team on 12 October 2026. This checklist covers the proposal, not the final-project grading rubric.

| Requirement | Prepared material | Status |
|---|---|---|
| Approximately 5-7 minutes | Recording guide: target 6:20, 930 narration words | Prepared; verify actual duration in rehearsal |
| Algorithm overview, 1-2 minutes | Slides 2-3: 1:40 | Ready |
| Assigned algorithm and classification | Slide 2: IQL, offline model-free actor-critic | Ready |
| Main idea and update mechanism | Slides 2-3: expectile value, twin critics, advantage-weighted actor | Ready |
| Environment description, 2-3 minutes | Slides 4-6: 2:40 | Ready |
| Observation/state space | Slide 4: 17 observed values, absolute x excluded | Ready |
| Action space | Slide 4: six continuous joint torques in [-1, 1] | Ready |
| Reward and agent goal | Slide 5: forward velocity minus effort cost | Ready |
| Episode termination conditions | Slide 5: no natural termination, 1,000-step truncation | Ready |
| Expected learned behavior | Slide 5: coordinated gait and sustained forward motion | Ready |
| Motivation, 1-2 minutes | Slides 7-8: 1:40 | Ready |
| Why this environment and algorithm | Slide 7: continuous control, available offline data, measurable behavior | Ready |
| Advantages, challenges and interesting aspects | Slides 7-8: advantage weighting, coverage, seed failures, physics versions | Ready |
| Compatible trainable/evaluable combination | Slide 6: Minari dataset and dataset-matched environment | Ready |
| Clear later training/evaluation plan | Slides 6 and 8: three seeds, 500k updates, selection and final test episodes | Ready |
| Presentation file in .pptx or .pdf | `IQL_RL_Proposal.pptx` and `IQL_RL_Proposal.pdf` | Ready; submit either format |
| Video with slides and verbal explanation | Follow `RECORDING_GUIDE.md` | Team must record |
| Presentation and recording in one ZIP | `make_submission_zip.ps1` or manual instructions | Create after recording |
| Upload ZIP to Moodle | Final checklist in recording guide | Team must upload |

The handout does not require final training results in the proposal. The deck focuses on the algorithm, task, suitability and project protocol. The repository already contains implemented training/evaluation and separate experiment evidence; this proposal does not imply those modules are unfinished.

An environment rollout MP4 is not the required narrated presentation video. Do not substitute it for your screen recording. The preparation materials ZIP is not the final Moodle submission because it contains no narrated recording.

## Sources

- [IQL paper](https://arxiv.org/abs/2110.06169)
- [Official IQL implementation](https://github.com/ikostrikov/implicit_q_learning)
- [HalfCheetah documentation](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)
- [Minari medium dataset](https://minari.farama.org/datasets/mujoco/halfcheetah/medium-v0/)
- Project implementation and configuration at commit `c372ec0423b81839956ec00dd7338700db8d3737`.

Technical source URLs and the environment image provenance are also embedded in the PowerPoint speaker notes.
