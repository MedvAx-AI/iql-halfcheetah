# Screen recording guide: IQL proposal

Use `IQL_RL_Proposal.pptx` in PowerPoint or import it into Google Slides. The PDF is a fallback for displaying the same slides. This is a proposal presentation, so you can satisfy the assignment entirely with slides and spoken explanation.

Target duration: **6 minutes 20 seconds**. The script has 930 words, approximately 147 words per minute at that target. Rehearse with a timer and aim for 6:20-6:45, staying within the required 5-7 minutes. Actual duration depends on your speaking pace and pauses. If you speak slowly, remove repeated explanations rather than rush the technical terms.

The suggested speakers reflect project roles. One person can read the entire script if your group chooses. The handout does not require all five people to speak. If switching speakers, include the handovers in the seven-minute limit.

## Before recording

1. Open the deck and review all eight slides. Use the speaker notes or this guide on a second screen, on paper, or in presenter view.
2. Open Zoom or your screen recorder. Select the slide-show window or presentation screen and enable microphone audio. The handout requires slides and verbal explanation; it does not specify a webcam requirement.
3. Make a short audio-and-screen test. Play it back and check that the narration is audible and the slide text is readable. Share the slide-show screen rather than presenter notes.
4. Rehearse once with a timer. Start on slide 1. Advance through the slides at the times below. Do not run training or notebook installation during this recording.
5. Start the actual recording, enter full-screen slide-show mode, and speak the narration below. Keep the cursor still except when pointing to an item.

## Timing and what to show

| Slide | Time | Suggested speaker | Topic |
|---|---|---|---|
| 1 | 0:00-0:20 | Aleksandr Medvedev | Implicit Q-Learning on HalfCheetah |
| 2 | 0:20-1:05 | Leo Vesin | IQL: learning from a fixed dataset |
| 3 | 1:05-2:00 | Leo Vesin | The IQL update |
| 4 | 2:00-3:05 | Nikita Shankin | HalfCheetah-v5: observation and action |
| 5 | 3:05-4:00 | Nikita Shankin | Reward, goal and episode ending |
| 6 | 4:00-4:40 | Ruslan Nasibullin | Offline data and evaluation protocol |
| 7 | 4:40-5:30 | Telman Nuruzov | Why IQL and HalfCheetah fit |
| 8 | 5:30-6:20 | Aleksandr Medvedev | Challenges and final-project plan |

Section totals: introduction 0:20; algorithm slides 2-3 1:40; environment/protocol slides 4-6 2:40; motivation/plan slides 7-8 1:40.

## Step-by-step narration

Read the text under **Say**. The **Show** instructions are actions, not narration.

### Step 1: slide 1, 0:00-0:20

**Show:** Show the title slide in full-screen presentation mode. Keep it visible while introducing the project and team.

**Say:**

Hello. We are presenting our reinforcement learning project proposal: Implicit Q-Learning on HalfCheetah. Our team is Aleksandr Medvedev, Nikita Shankin, Leo Vesin, Ruslan Nasibullin, and Telman Nuruzov. We will explain the algorithm, describe the environment, and show why this combination fits our project.

**Advance:** Change to slide 2 near 0:20.

### Step 2: slide 2, 0:20-1:05

**Show:** Show slide 2. Point to Classification, then Training data, then Main idea. Explain offline training versus simulator evaluation.

**Say:**

Our assigned algorithm is Implicit Q-Learning, usually called IQL. It is an offline, model-free actor-critic method. Offline means that training uses a fixed collection of recorded transitions rather than collecting new actions from the simulator. Each transition includes an observation, an action, a reward, the next observation, and episode flags. The critic estimates how useful actions are. The actor learns a continuous control policy from those recorded actions. IQL first learns state values that emphasize better actions present in the dataset. It then gives actions with higher estimated advantage more influence during policy training. We use simulator interactions to evaluate the learned policy, while keeping training itself offline.

**Advance:** Change to slide 3 near 1:05.

### Step 3: slide 3, 1:05-2:00

**Show:** Show slide 3. Follow the four rows from top to bottom. Explain the equations in words; do not read every symbol.

**Say:**

The implementation has a state-value network, two Q networks, and a policy network. The value network uses expectile regression. With an expectile of zero point seven, positive errors receive more weight, so the value estimate favors stronger actions in the dataset. The Q networks learn from the recorded reward plus the discounted value of the next state. We use a discount of zero point nine nine and the minimum of the target critics when estimating advantage. The policy then maximizes the likelihood of recorded actions, weighted by the exponential of three times their advantage, capped at one hundred. Three is our inverse-temperature setting. A tanh-squashed Gaussian keeps actions within the environment bounds. These updates avoid an explicit maximization over unseen actions in the value backup, although limited dataset coverage remains a challenge.

**Advance:** Change to slide 4 near 2:00.

### Step 4: slide 4, 2:00-3:05

**Show:** Show slide 4. Point to the robot, then the 17-value observation and 6-value action descriptions. The image is sufficient; no live notebook run is needed.

**Say:**

We selected HalfCheetah version five, a two-dimensional locomotion environment implemented in Gymnasium with MuJoCo physics. The image shows the actual environment from our project. The agent controls the robot's front and back legs and tries to move forward. Its observation is a vector of seventeen values. Eight describe positions or angles, including body height, body orientation, and leg joint angles. The other nine describe velocities. The default observation excludes absolute horizontal position, which encourages a policy that can run from different locations. The action is a vector of six continuous torque controls for the front and back thigh, shin, and foot joints. Each component is bounded between minus one and one. This is a continuous-control problem: at every step, the policy must coordinate the joints using the current pose and motion, rather than choose from a small set of discrete moves.

**Advance:** Change to slide 5 near 3:05.

### Step 5: slide 5, 3:05-4:00

**Show:** Show slide 5. Point to the reward first, then Episode ending. Explicitly say that falling does not end HalfCheetah.

**Say:**

The reward combines forward progress with an effort penalty. With the default settings, it equals horizontal forward velocity minus zero point one times the sum of squared action components. Running forward increases reward, while unnecessarily large torques reduce it. The agent should learn a coordinated gait that sustains speed efficiently throughout the episode. HalfCheetah has an important episode rule: a fall does not naturally terminate the task. The standard environment continues until the time limit of one thousand control steps, which corresponds to about fifty simulated seconds. Therefore, a policy can fall and remain on the ground for the rest of the rollout. We assess the complete episode, including that behavior. In our training targets, a time-limit truncation preserves the next-state value term. Only a true termination would remove that bootstrap term. This distinction keeps the learning target consistent with the task.

**Advance:** Change to slide 6 near 4:00.

### Step 6: slide 6, 4:00-4:40

**Show:** Show slide 6. Explain the fixed dataset, planned budget and repeatable evaluation. Do not switch to Colab or wait for training.

**Say:**

For offline training, we use Minari's HalfCheetah medium dataset. It contains one thousand episodes and one million transitions. Its recorded continuous actions and observations match the chosen task. Our project protocol uses five hundred thousand offline updates for each of three training seeds, with batches of two hundred fifty-six transitions. We evaluate intermediate checkpoints over ten episodes per run and compare raw episodic return with a uniform random policy. We report variation across episodes separately from variation across training seeds. Checkpoint selection and final testing use separate episode seeds. We also keep full rollouts and match the dataset environment with MuJoCo version three point two point three.

**Advance:** Change to slide 7 near 4:40.

### Step 7: slide 7, 4:40-5:30

**Show:** Show slide 7. Connect each reason to the algorithm or environment. Emphasize that better performance must be measured.

**Say:**

We chose this environment because it gives IQL a concrete and measurable continuous-control task. The policy can directly represent the bounded torques needed by the robot, so we do not need to discretize the action space. A ready-made offline dataset makes the project feasible without an additional data-collection stage. It contains recorded locomotion behavior from which IQL can learn state values and action advantages. The advantage weights let stronger recorded actions influence the learned policy more heavily. This creates a useful research question: can the resulting policy improve on the recorded behavior while remaining reliable across episodes? Improvement is an empirical hypothesis, not a guarantee. HalfCheetah also provides numerical returns and clear visual behavior. These let us compare policies quantitatively and inspect whether the robot maintains a useful gait throughout a full rollout.

**Advance:** Change to slide 8 near 5:30.

### Step 8: slide 8, 5:30-6:20

**Show:** Show slide 8. Explain the challenges, name the responsibility owners, and finish. Leave the slide visible briefly before stopping the recording.

**Say:**

The main challenges are limited dataset coverage and variation between trained policies. A high average return can hide occasional falls, so our final evaluation includes multiple seeds and complete videos. Physics versions can also change a rollout, which is why we pin the environment dependencies and document evaluation settings. Our final-project workflow is to validate the dataset, train IQL, compare checkpoints on selection episodes, and test the selected policy on separate final episodes. Aleksandr handles integration and reproducibility. Nikita owns the environment and dataset. Leo owns IQL. Ruslan owns evaluation, and Telman owns the notebook and presentation. The shared deliverables are a reproducible Colab, saved checkpoints, evaluation plots, and a complete policy demonstration. This proposal establishes our algorithm, task, and evaluation approach. Thank you.

**Finish:** Pause briefly on the last slide, then stop the recording.

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
