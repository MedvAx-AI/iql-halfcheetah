# Visual proposal slide source

The PPTX contains editable diagrams and charts. This source records each slideâ€™s purpose and concise narration.

## Slide 1: Implicit Q-Learning on HalfCheetah

Introduce the title and team. Point to the actual HalfCheetah rollout image; hold briefly before advancing.

Narration (20 seconds including pauses):

Our project uses Implicit Q-Learning, or IQL, to learn continuous locomotion from recorded data. We will explain how the algorithm works, what HalfCheetah requires, and why this combination is suitable. The final slide shows measured feasibility evidence from our existing project.

## Slide 2: Offline data, learned values, continuous control

Follow the upper diagram from recorded transitions to learner to policy. Point to the separate simulator evaluation path. Pause so the audience can follow the distinction.

Narration (50 seconds including pauses):

IQL is offline, model-free actor-critic reinforcement learning. Training starts with a fixed dataset of observations, actions, rewards and next observations. The diagram separates learning from evaluation. IQL estimates values using recorded actions, then trains the actor to imitate stronger actions more heavily. It avoids explicitly maximizing the value backup over unseen actions. The simulator is used to evaluate the policy after training, rather than collect new training data. The policy produces six bounded torque controls at each step.

## Slide 3: Three updates inside IQL

Explain each update in order. Point to 0.7, 0.99 and the weight cap of 100. Read the equations in words and allow brief pauses.

Narration (50 seconds including pauses):

The learner has three updates. First, the state-value network uses an upper expectile of zero point seven, emphasizing higher-valued recorded actions. Second, the twin Q networks fit reward plus discounted next-state value, with discount zero point nine nine. Third, the actor fits recorded actions using advantage weights: the exponential of three times advantage, capped at one hundred. Advantage is the minimum target-critic estimate minus the state value. Three is inverse temperature. A tanh-squashed Gaussian keeps the actor's outputs within the action bounds. The repeated updates improve the policy without changing the fixed dataset.

## Slide 4: 17 observations, 6 joint controls

Point to the actual robot, then trace the 8 plus 9 observation split. Explain the three joints on each of two legs and the action bound. Leave time for the audience to inspect the image.

Narration (65 seconds including pauses):

HalfCheetah version five is a two-dimensional locomotion task in Gymnasium and MuJoCo. The robot has front and back legs. Its seventeen-value observation combines eight position or angle values with nine velocities. These describe the robot's pose and motion. Absolute horizontal position is excluded from the default observation. The action has six continuous torque components: thigh, shin and foot on each leg. Every component lies between minus one and one. The policy must use the current observation to coordinate these joints into a useful gait. The image comes from a real project rollout, while the color split shows the observation structure.

## Slide 5: Forward motion earns reward

Follow the numerical reward example from speed through effort cost to net reward. Trace the 0-to-50-second episode line. Emphasize that a fall does not end the episode.

Narration (55 seconds including pauses):

The default reward is forward velocity minus zero point one times squared control effort. The numerical example is illustrative: at five meters per second, six action components of zero point five produce a cost of zero point fifteen, so the reward is four point eight five. This encourages efficient forward motion. The agent should learn a gait that maintains speed throughout the episode. A fall does not terminate HalfCheetah. The time limit is one thousand steps, about fifty simulated seconds. A failed policy can remain on the ground until that limit. Time-limit truncation still preserves next-state value bootstrapping in our learning targets.

## Slide 6: A reproducible training and evaluation plan

Point to the large dataset and budget numbers, then follow the training-to-selection-to-final-test timeline. Distinguish training seeds from evaluation episode seeds.

Narration (40 seconds including pauses):

We use Minari's HalfCheetah medium dataset: one million transitions from one thousand episodes. Our protocol trains three seeds for five hundred thousand offline updates each, with batches of two hundred fifty-six. We evaluate checkpoints over ten episodes, compare raw return with a random baseline, then test the selected checkpoint on separate final episode seeds. We retain full videos and separate episode variation from training-seed variation. MuJoCo three point two point three matches the recorded dataset runtime.

## Slide 7: Why this algorithm fits this task

Follow the three algorithm-to-task matches. Explain the challenges along the bottom. Point to the team ownership line and name each personâ€™s contribution without reading all slide labels aloud.

Narration (60 seconds including pauses):

The match has three practical reasons. Continuous joint torques fit a continuous actor directly. Available offline locomotion data makes training feasible without a new data-collection stage. Finally, numerical returns and visible movement make evaluation understandable. IQL's advantage weighting can favor better recorded actions, but improvement is something to test, not assume. The main challenges are limited coverage, policy variation and simulator differences. We address them with multiple training seeds, pinned dependencies and full-episode evaluation. Aleksandr owns integration, Nikita the environment and data, Leo IQL, Ruslan evaluation, and Telman the notebook and presentation. Our shared outputs are a reproducible Colab, checkpoints, plots and complete rollouts.

## Slide 8: Learning progress and final test

Trace all three measured curves. Point to the separate final-test return and stationary-tail counts. Say the other-seed limitation clearly, then pause on the closing slide.

Narration (50 seconds including pauses):

This chart shows recorded Colab validation means for all three training seeds at five measured checkpoints. Each point averages ten episodes. Performance varies across seeds, and the lines do not claim values between measurements. The selected seed-one policy reached a raw return of about fifteen thousand two hundred sixty on ten separate final-test episodes, compared with about two thousand twenty-six at one hundred thousand updates. Long stationary tails decreased from ten out of ten episodes to zero out of ten. Those tails mean at least five seconds of very slow movement at the end. Other training seeds still fail, and these samples do not guarantee success everywhere. The evidence supports feasibility while preserving the limits. Thank you.
