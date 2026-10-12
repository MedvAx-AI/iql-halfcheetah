# Proposal requirements coverage

Source: the two-page course handout `RL_Project_Proposal_Presentation.pdf` supplied by the team on 12 October 2026. This checklist covers the proposal, not the final-project grading rubric.

| Requirement | Prepared material | Status |
|---|---|---|
| Approximately 5-7 minutes | Recording guide: target 5:00, 370-word prompts plus visual explanation | Prepared; verify actual duration in rehearsal |
| Algorithm overview, 1-2 minutes | Slides 2-3: 1:20 | Ready |
| Assigned algorithm and classification | Slide 2: IQL, offline model-free actor-critic | Ready |
| Main idea and update mechanism | Slides 2-3: expectile value, twin critics, advantage-weighted actor | Ready |
| Environment description, 2-3 minutes | Slides 4-5: 2:00, plus slide 6 protocol: 0:20 | Ready |
| Observation/state space | Slide 4: 17 observed values, absolute x excluded | Ready |
| Action space | Slide 4: six continuous joint torques in [-1, 1] | Ready |
| Reward and agent goal | Slide 5: forward velocity minus effort cost | Ready |
| Episode termination conditions | Slide 5: no natural termination, 1,000-step truncation | Ready |
| Expected learned behavior | Slide 5: coordinated gait and sustained forward motion | Ready |
| Motivation, 1-2 minutes | Slides 7-8: 1:10, including optional feasibility evidence | Ready |
| Why this environment and algorithm | Slide 7: continuous control, available offline data, measurable behavior | Ready |
| Advantages, challenges and interesting aspects | Slides 7-8: advantage weighting, coverage, seed failures, physics versions | Ready |
| Compatible trainable/evaluable combination | Slide 6: Minari dataset and dataset-matched environment | Ready |
| Clear later training/evaluation plan | Slides 6 and 8: three seeds, 500k updates, selection and final test episodes | Ready |
| Presentation file in .pptx or .pdf | `IQL_RL_Proposal.pptx` and `IQL_RL_Proposal.pdf` | Ready; submit either format |
| Video with slides and verbal explanation | Follow `RECORDING_GUIDE.md` | Team must record |
| Presentation and recording in one ZIP | `make_submission_zip.ps1` or manual instructions | Create after recording |
| Upload ZIP to Moodle | Final checklist in recording guide | Team must upload |

The handout does not require final training results in the proposal. The visual revision retains the required algorithm, environment, motivation and protocol, and adds measured feasibility evidence to meet the team's request for graphs and metrics. The repository already contains implemented training/evaluation; this proposal does not imply those modules are unfinished.

## Visual evidence

- Slide 4's editable doughnut has 8 pose dimensions and 9 velocities, totaling 17 observed dimensions.
- Slide 5's example uses forward velocity 5 m/s and six actions of 0.5: control cost is `0.1 * 6 * 0.5^2 = 0.15`, and reward is `5 - 0.15 = 4.85`. It is explicitly illustrative.
- Slide 8's editable scatter chart uses the measured validation means for all three training seeds at 100k, 150k, 200k, 300k and 500k updates, with correct numerical x-axis spacing. Data are rounded to two decimals for chart workbook portability. The source is `docs/evidence/continuation/validation/summary.json`. Each point averages 10 episodes; connecting lines do not estimate unmeasured checkpoints. The graph shows means without uncertainty bands.
- The separate final-test metric is training seed 1 at 500k: raw mean 15,260.36 versus paired 100k mean 2,026.01 on 10 independent episodes, seeds 30000-30009. Long stationary tails fall from 10/10 to 0/10. The source is `docs/evidence/continuation/independent-final-test.json`. A tail is at least 5 seconds at the end with absolute forward velocity at most 0.1 m/s. This is a descriptive diagnostic, not a standard benchmark success score. These samples do not guarantee universal performance, and other training seeds still fail.

An environment rollout MP4 is not the required narrated presentation video. Do not substitute it for your screen recording. The preparation materials ZIP is not the final Moodle submission because it contains no narrated recording.

## Sources

- [IQL paper](https://arxiv.org/abs/2110.06169)
- [Official IQL implementation](https://github.com/ikostrikov/implicit_q_learning)
- [HalfCheetah documentation](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)
- [Minari medium dataset](https://minari.farama.org/datasets/mujoco/halfcheetah/medium-v0/)
- Project implementation and configuration at commit `c372ec0423b81839956ec00dd7338700db8d3737`.

PowerPoint speaker notes cite numbered references. Full source URLs and environment image provenance are in `SLIDE_SOURCES.md`.
