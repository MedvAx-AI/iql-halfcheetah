# Recording guide: visual IQL proposal

Target: **6:30**. The shortened narration has **711 words**, down from 930. Use a natural pace and the suggested pauses to let the audience read the diagrams. At about 130 words per minute, narration takes roughly 5:28; pauses and slide changes bring it near 6:30. Rehearse once and keep the actual video within 5-7 minutes.

Show the full-screen slides and speak the text under **Say**. Instructions under **Show** are actions, not narration. The charts are recorded project evidence, not measurements from a fresh notebook run. Numerical reward values on slide 5 are explicitly an illustrative example.

Suggested speakers reflect the project roles. One person can deliver all slides. Speaker changes count toward the total time.

## Before recording

1. Open the deck and review all eight slides. Use the speaker notes or this guide on a second screen, on paper, or in presenter view.
2. Open Zoom or your screen recorder. Select the slide-show window or presentation screen and enable microphone audio. The handout requires slides and verbal explanation; it does not specify a webcam requirement.
3. Make a short audio-and-screen test. Play it back and check that the narration is audible and the slide text is readable. Share the slide-show screen rather than presenter notes.
4. Rehearse once with a timer. Start on slide 1. Advance through the slides at the times below. Do not run training or notebook installation during this recording.
5. Start the actual recording, enter full-screen slide-show mode, and speak the narration below. Keep the cursor still except when pointing to an item.



## Timing and what to show

| Slide | Time | Speaker | Visual |
|---|---|---|---|
| 1 | 0:00-0:20 | Aleksandr Medvedev | Implicit Q-Learning on HalfCheetah |
| 2 | 0:20-1:10 | Leo Vesin | Offline data, learned values, continuous control |
| 3 | 1:10-2:00 | Leo Vesin | Three updates inside IQL |
| 4 | 2:00-3:05 | Nikita Shankin | 17 observations, 6 joint controls |
| 5 | 3:05-4:00 | Nikita Shankin | Forward motion earns reward |
| 6 | 4:00-4:40 | Ruslan Nasibullin | A reproducible training and evaluation plan |
| 7 | 4:40-5:40 | Telman Nuruzov | Why this algorithm fits this task |
| 8 | 5:40-6:30 | Aleksandr Medvedev | Learning progress and final test |

Section totals: introduction 0:20; algorithm slides 2-3 1:40; environment and protocol slides 4-6 2:40; motivation and feasibility slides 7-8 1:50.

Final training results are optional in the proposal. Slide 8 adds already-measured feasibility evidence because this revision includes metrics and graphs. It does not replace any required proposal topic.

## Step-by-step narration

### Slide 1: Implicit Q-Learning on HalfCheetah (0:00-0:20)

**Show:** Introduce the title and team. Point to the actual HalfCheetah rollout image; hold briefly before advancing.

**Say:**

Our project uses Implicit Q-Learning, or IQL, to learn continuous locomotion from recorded data. We will explain how the algorithm works, what HalfCheetah requires, and why this combination is suitable. The final slide shows measured feasibility evidence from our existing project.

**Timing:** Allow approximately 1 seconds for pointing, pauses and changing slides. Advance near 0:20.

### Slide 2: Offline data, learned values, continuous control (0:20-1:10)

**Show:** Follow the upper diagram from recorded transitions to learner to policy. Point to the separate simulator evaluation path. Pause so the audience can follow the distinction.

**Say:**

IQL is offline, model-free actor-critic reinforcement learning. Training starts with a fixed dataset of observations, actions, rewards and next observations. The diagram separates learning from evaluation. IQL estimates values using recorded actions, then trains the actor to imitate stronger actions more heavily. It avoids explicitly maximizing the value backup over unseen actions. The simulator is used to evaluate the policy after training, rather than collect new training data. The policy produces six bounded torque controls at each step.

**Timing:** Allow approximately 14 seconds for pointing, pauses and changing slides. Advance near 1:10.

### Slide 3: Three updates inside IQL (1:10-2:00)

**Show:** Explain each update in order. Point to 0.7, 0.99 and the weight cap of 100. Read the equations in words and allow brief pauses.

**Say:**

The learner has three updates. First, the state-value network uses an upper expectile of zero point seven, emphasizing higher-valued recorded actions. Second, the twin Q networks fit reward plus discounted next-state value, with discount zero point nine nine. Third, the actor fits recorded actions using advantage weights: the exponential of three times advantage, capped at one hundred. Advantage is the minimum target-critic estimate minus the state value. Three is inverse temperature. A tanh-squashed Gaussian keeps the actor's outputs within the action bounds. The repeated updates improve the policy without changing the fixed dataset.

**Timing:** Allow approximately 7 seconds for pointing, pauses and changing slides. Advance near 2:00.

### Slide 4: 17 observations, 6 joint controls (2:00-3:05)

**Show:** Point to the actual robot, then trace the 8 plus 9 observation split. Explain the three joints on each of two legs and the action bound. Leave time for the audience to inspect the image.

**Say:**

HalfCheetah version five is a two-dimensional locomotion task in Gymnasium and MuJoCo. The robot has front and back legs. Its seventeen-value observation combines eight position or angle values with nine velocities. These describe the robot's pose and motion. Absolute horizontal position is excluded from the default observation. The action has six continuous torque components: thigh, shin and foot on each leg. Every component lies between minus one and one. The policy must use the current observation to coordinate these joints into a useful gait. The image comes from a real project rollout, while the color split shows the observation structure.

**Timing:** Allow approximately 19 seconds for pointing, pauses and changing slides. Advance near 3:05.

### Slide 5: Forward motion earns reward (3:05-4:00)

**Show:** Follow the numerical reward example from speed through effort cost to net reward. Trace the 0-to-50-second episode line. Emphasize that a fall does not end the episode.

**Say:**

The default reward is forward velocity minus zero point one times squared control effort. The numerical example is illustrative: at five meters per second, six action components of zero point five produce a cost of zero point fifteen, so the reward is four point eight five. This encourages efficient forward motion. The agent should learn a gait that maintains speed throughout the episode. A fall does not terminate HalfCheetah. The time limit is one thousand steps, about fifty simulated seconds. A failed policy can remain on the ground until that limit. Time-limit truncation still preserves next-state value bootstrapping in our learning targets.

**Timing:** Allow approximately 8 seconds for pointing, pauses and changing slides. Advance near 4:00.

### Slide 6: A reproducible training and evaluation plan (4:00-4:40)

**Show:** Point to the large dataset and budget numbers, then follow the training-to-selection-to-final-test timeline. Distinguish training seeds from evaluation episode seeds.

**Say:**

We use Minari's HalfCheetah medium dataset: one million transitions from one thousand episodes. Our protocol trains three seeds for five hundred thousand offline updates each, with batches of two hundred fifty-six. We evaluate checkpoints over ten episodes, compare raw return with a random baseline, then test the selected checkpoint on separate final episode seeds. We retain full videos and separate episode variation from training-seed variation. MuJoCo three point two point three matches the recorded dataset runtime.

**Timing:** Allow approximately 5 seconds for pointing, pauses and changing slides. Advance near 4:40.

### Slide 7: Why this algorithm fits this task (4:40-5:40)

**Show:** Follow the three algorithm-to-task matches. Explain the challenges along the bottom. Point to the team ownership line and name each personâ€™s contribution without reading all slide labels aloud.

**Say:**

The match has three practical reasons. Continuous joint torques fit a continuous actor directly. Available offline locomotion data makes training feasible without a new data-collection stage. Finally, numerical returns and visible movement make evaluation understandable. IQL's advantage weighting can favor better recorded actions, but improvement is something to test, not assume. The main challenges are limited coverage, policy variation and simulator differences. We address them with multiple training seeds, pinned dependencies and full-episode evaluation. Aleksandr owns integration, Nikita the environment and data, Leo IQL, Ruslan evaluation, and Telman the notebook and presentation. Our shared outputs are a reproducible Colab, checkpoints, plots and complete rollouts.

**Timing:** Allow approximately 12 seconds for pointing, pauses and changing slides. Advance near 5:40.

### Slide 8: Learning progress and final test (5:40-6:30)

**Show:** Trace all three measured curves. Point to the separate final-test return and stationary-tail counts. Say the other-seed limitation clearly, then pause on the closing slide.

**Say:**

This chart shows recorded Colab validation means for all three training seeds at five measured checkpoints. Each point averages ten episodes. Performance varies across seeds, and the lines do not claim values between measurements. The selected seed-one policy reached a raw return of about fifteen thousand two hundred sixty on ten separate final-test episodes, compared with about two thousand twenty-six at one hundred thousand updates. Long stationary tails decreased from ten out of ten episodes to zero out of ten. Those tails mean at least five seconds of very slow movement at the end. Other training seeds still fail, and these samples do not guarantee success everywhere. The evidence supports feasibility while preserving the limits. Thank you.

**Finish:** Let the audience inspect the graph, state the limitation, then stop recording.

## After recording

1. Save/export the recording as `IQL_Proposal_Recording.mp4` if possible.
2. Play the whole recording. Check that all eight slides and the spoken explanation appear, the sound is clear, and the duration is 5-7 minutes.
3. Create a folder named with your assigned group name, for example `YourGroup_RL_Proposal`. Put **one** presentation file (`IQL_RL_Proposal.pptx` or `.pdf`) and your actual narrated recording inside it.
4. Compress those two files into `YourGroup_RL_Proposal.zip`. If your group has no assigned label, `IQL_HalfCheetah_RL_Proposal.zip` is a descriptive default. The handout gives a suggested naming pattern, not an assigned group number.
5. Open the ZIP and confirm both files are present. Upload the ZIP to the correct Moodle assignment and check its submission confirmation.

### Optional Windows ZIP helper

Keep `make_submission_zip.ps1` beside the PPTX and PDF. Open PowerShell in that folder and run:

```powershell
.\make_submission_zip.ps1 -RecordingPath "C:\path\IQL_Proposal_Recording.mp4" -GroupName "YourGroup"
```

Replace the example recording path and group name with yours. Add `-PresentationFormat pdf` if you choose the PDF. The helper refuses to overwrite an existing ZIP. It checks that a non-empty video file exists; you must still check its content and duration.

The materials archive contains preparation files only. Do not submit it unchanged. A standalone HalfCheetah rollout video does not replace the required narrated presentation recording.

## Useful links

- [Project repository](https://github.com/MedvAx-AI/iql-halfcheetah)
- [Colab notebook](https://colab.research.google.com/github/MedvAx-AI/iql-halfcheetah/blob/main/notebooks/iql_halfcheetah.ipynb)
- [Requirements coverage](REQUIREMENTS_CHECKLIST.md)


