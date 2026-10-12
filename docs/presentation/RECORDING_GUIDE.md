# Five-minute recording guide

**Target: 5:00.** The short narration is 370 words, down from 711. Use the notes as prompts: explain and point to the visuals within each slide’s allotted time. Reading the text alone may finish early. Rehearse with a timer and follow the slide-change schedule rather than rushing. The schedule includes visual explanation and transitions.

Sections: introduction 0:10; algorithm 1:20; environment and protocol 2:20; motivation and feasibility 1:10. The environment description itself has two full minutes on slides 4-5.

## Before recording

1. Open the PPTX in slide-show mode and keep this guide or presenter notes beside you.
2. Select the slide-show window in your recorder and enable microphone audio. Make a short test and check playback.
3. Rehearse once. One presenter can deliver all slides; any speaker changes must fit within five minutes.
4. Record the slides and spoken explanation. You do not need to run Colab or training during the presentation.

## Slide schedule

| Slide | Time | Suggested speaker |
|---|---|---|
| 1 | 0:00-0:10 | Aleksandr Medvedev |
| 2 | 0:10-0:45 | Leo Vesin |
| 3 | 0:45-1:30 | Leo Vesin |
| 4 | 1:30-2:30 | Nikita Shankin |
| 5 | 2:30-3:30 | Nikita Shankin |
| 6 | 3:30-3:50 | Ruslan Nasibullin |
| 7 | 3:50-4:30 | Telman Nuruzov |
| 8 | 4:30-5:00 | Aleksandr Medvedev |

## Short narration and what to show

### Slide 1: Implicit Q-Learning (0:00-0:10)

**Show:** Introduce the title and team. Advance at 0:10.

**Say:**

Our project uses Implicit Q-Learning to teach HalfCheetah efficient locomotion from recorded data.

### Slide 2: How IQL learns from recorded data (0:10-0:45)

**Show:** Trace data, learner, policy, then the separate evaluation path.

**Say:**

IQL is offline, model-free actor-critic learning. It trains on fixed transitions, then gives better recorded actions more weight when fitting the policy. The actor outputs six continuous torques. Simulator rollouts evaluate the learned policy, without collecting new training data.

### Slide 3: The three IQL updates (0:45-1:30)

**Show:** Point to each update and its highlighted parameter. Explain the formulas in words.

**Say:**

IQL updates three components. The value network uses expectile 0.7 to favor stronger recorded actions. Twin critics fit reward plus discounted next-state value, with discount 0.99. The actor imitates actions using advantage weights, capped at 100. Advantage is minimum target Q minus V. The weight uses inverse temperature 3. A tanh-squashed Gaussian keeps actions bounded.

### Slide 4: HalfCheetah observations and controls (1:30-2:30)

**Show:** Point to pose versus velocity, then identify each leg and its three joints. Use the image to explain the observation-action relationship.

**Say:**

HalfCheetah-v5 is a two-dimensional locomotion task in Gymnasium and MuJoCo. Its 17 observations contain eight pose values and nine velocities. Absolute horizontal position is excluded. Six actions control the thigh, shin and foot on both legs. Each torque lies between minus one and one. The agent must coordinate these joints into a sustained forward gait.

### Slide 5: Reward and episode length (2:30-3:30)

**Show:** Explain the example, point to speed and effort, then trace the full episode line. Emphasize the difference between a fall and the time limit.

**Say:**

Reward is forward velocity minus 0.1 times squared action effort. In this example, speed contributes 5 and control costs 0.15, giving 4.85 reward. The goal is efficient, sustained forward motion. A fall does not end HalfCheetah: the episode lasts 1,000 steps, about 50 simulated seconds. Time-limit truncation preserves next-state value bootstrapping. We inspect complete episodes to detect falls and stalls.

### Slide 6: Training and evaluation (3:30-3:50)

**Show:** Point to the three large numbers and trace train, select, final test.

**Say:**

We use one million Minari transitions. Three training seeds receive 500k updates each. Checkpoints use ten evaluation episodes and a random baseline. The selected policy then uses separate final-test seeds. We retain full videos and pin MuJoCo 3.2.3 for reproducibility.

### Slide 7: Why HalfCheetah fits IQL (3:50-4:30)

**Show:** Connect each environment property to IQL. Point to the risks and team ownership line.

**Say:**

HalfCheetah fits IQL because it has continuous controls, available offline data, and measurable locomotion. Advantage weighting can favor better actions, but improvement must be tested. Main challenges are data coverage, seed failures and physics differences. We address them with multiple seeds, pinned dependencies and full-episode evaluation. Team responsibilities and deliverables appear here.

### Slide 8: Learning curves and final test (4:30-5:00)

**Show:** Trace the measured curves, point to the independent test metrics, and state the limitation.

**Say:**

The curves show measured means for all three training seeds, with ten episodes per point. On ten separate final-test episodes, selected seed 1 achieved about 15,260 return versus 2,026 at 100k. Long stationary tails decreased from ten episodes to zero. Other training seeds still fail, so these results demonstrate feasibility rather than guaranteed success. Thank you.

## After recording

1. Play the recording and confirm that all slides and clear narration appear. Aim for five minutes and avoid finishing under the handout’s five-minute minimum.
2. Put the PPTX or PDF and your actual narrated recording into one ZIP. The preparation materials ZIP is not the final submission.
3. Upload the final ZIP to Moodle and check the confirmation.

Optional Windows helper (keep it beside the slide files):

```powershell
.\make_submission_zip.ps1 -RecordingPath "C:\path\IQL_Proposal_Recording.mp4" -GroupName "YourGroup"
```

Replace the example path and group name. Add `-PresentationFormat pdf` to use PDF. The helper does not check narration or duration, so check the video yourself.

Technical references and measurement definitions: [SLIDE_SOURCES.md](SLIDE_SOURCES.md). Sources are not part of the spoken script.
