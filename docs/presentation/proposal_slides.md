# Five-minute visual proposal

Visual slides are unchanged. Speaker notes use the following short narration and timing. Full references are in `SLIDE_SOURCES.md`.

## Slide 1: Implicit Q-Learning on HalfCheetah

Time: 0:00-0:10. Sources: [1], [4].

Our project uses Implicit Q-Learning to teach HalfCheetah efficient locomotion from recorded data.

## Slide 2: Offline data, learned values, continuous control

Time: 0:10-0:45. Sources: [1].

IQL is offline, model-free actor-critic learning. It trains on fixed transitions, then gives better recorded actions more weight when fitting the policy. The actor outputs six continuous torques. Simulator rollouts evaluate the learned policy, without collecting new training data.

## Slide 3: Three updates inside IQL

Time: 0:45-1:30. Sources: [1], [3].

IQL updates three components. The value network uses expectile 0.7 to favor stronger recorded actions. Twin critics fit reward plus discounted next-state value, with discount 0.99. The actor imitates actions using advantage weights, capped at 100. Advantage is minimum target Q minus V. The weight uses inverse temperature 3. A tanh-squashed Gaussian keeps actions bounded.

## Slide 4: 17 observations, 6 joint controls

Time: 1:30-2:30. Sources: [2], [4].

HalfCheetah-v5 is a two-dimensional locomotion task in Gymnasium and MuJoCo. Its 17 observations contain eight pose values and nine velocities. Absolute horizontal position is excluded. Six actions control the thigh, shin and foot on both legs. Each torque lies between minus one and one. The agent must coordinate these joints into a sustained forward gait.

## Slide 5: Forward motion earns reward

Time: 2:30-3:30. Sources: [2], [3].

Reward is forward velocity minus 0.1 times squared action effort. In this example, speed contributes 5 and control costs 0.15, giving 4.85 reward. The goal is efficient, sustained forward motion. A fall does not end HalfCheetah: the episode lasts 1,000 steps, about 50 simulated seconds. Time-limit truncation preserves next-state value bootstrapping. We inspect complete episodes to detect falls and stalls.

## Slide 6: A reproducible training and evaluation plan

Time: 3:30-3:50. Sources: [3], [5].

We use one million Minari transitions. Three training seeds receive 500k updates each. Checkpoints use ten evaluation episodes and a random baseline. The selected policy then uses separate final-test seeds. We retain full videos and pin MuJoCo 3.2.3 for reproducibility.

## Slide 7: Why this algorithm fits this task

Time: 3:50-4:30. Sources: [1], [2], [3].

HalfCheetah fits IQL because it has continuous controls, available offline data, and measurable locomotion. Advantage weighting can favor better actions, but improvement must be tested. Main challenges are data coverage, seed failures and physics differences. We address them with multiple seeds, pinned dependencies and full-episode evaluation. Team responsibilities and deliverables appear here.

## Slide 8: Learning progress and final test

Time: 4:30-5:00. Sources: [6].

The curves show measured means for all three training seeds, with ten episodes per point. On ten separate final-test episodes, selected seed 1 achieved about 15,260 return versus 2,026 at 100k. Long stationary tails decreased from ten episodes to zero. Other training seeds still fail, so these results demonstrate feasibility rather than guaranteed success. Thank you.
