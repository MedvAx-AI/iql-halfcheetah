# IQL proposal slide source

This source matches the eight-slide PPTX/PDF. Full narration is in `RECORDING_GUIDE.md` and PowerPoint speaker notes.

## Slide 1: Implicit Q-Learning on HalfCheetah

Suggested speaker: Aleksandr Medvedev. Time: 20 seconds.

RL project proposal

**Team:** Aleksandr Medvedev, Nikita Shankin, Leo Vesin Ruslan Nasibullin, Telman Nuruzov

Sources:

- https://github.com/MedvAx-AI/iql-halfcheetah

## Slide 2: IQL: learning from a fixed dataset

Suggested speaker: Leo Vesin. Time: 45 seconds.

**Classification:** Offline, model-free actor-critic RL

**Training data:** Recorded transitions: state, action, reward, next state

**Main idea:** Learn values from dataset actions, then give better actions more weight when fitting the policy

**Interaction:** Simulator rollouts evaluate the policy after training

Sources:

- https://arxiv.org/abs/2110.06169
- https://github.com/ikostrikov/implicit_q_learning

## Slide 3: The IQL update

Suggested speaker: Leo Vesin. Time: 55 seconds.

**State value V:** Upper expectile regression on dataset Q-values Expectile = 0.7

**Twin critics Q1, Q2:** Target = reward + 0.99 × (1 - terminated) × V(next state)

**Policy:** Advantage = min(target Q1, target Q2) - V Weighted log-likelihood of recorded actions

**Policy weight:** min(exp(3 × advantage), 100) Tanh-squashed Gaussian actions stay within [-1, 1]

Sources:

- https://arxiv.org/abs/2110.06169
- https://github.com/MedvAx-AI/iql-halfcheetah/blob/c372ec0423b81839956ec00dd7338700db8d3737/src/iql_project/iql.py
- https://github.com/MedvAx-AI/iql-halfcheetah/blob/c372ec0423b81839956ec00dd7338700db8d3737/configs/halfcheetah.toml

## Slide 4: HalfCheetah-v5: observation and action

Suggested speaker: Nikita Shankin. Time: 65 seconds.

**Environment:** 2D locomotion in Gymnasium and MuJoCo

**Observation: 17 values:** 8 position / angle values + 9 velocities Absolute horizontal position is excluded

**Action: 6 values:** Continuous torques for front and back leg joints Each action component lies in [-1, 1]

Sources:

- https://gymnasium.farama.org/environments/mujoco/half_cheetah/
- Image: project rollout, seed 1, 500k checkpoint, evaluation seed 10000, frame at 10 seconds. https://github.com/MedvAx-AI/iql-halfcheetah/releases/tag/policy-continuation-experiment

## Slide 5: Reward, goal and episode ending

Suggested speaker: Nikita Shankin. Time: 55 seconds.

**Reward:** Forward velocity - 0.1 × sum(action²)

**Agent goal:** Sustained forward motion with efficient torque use

**Episode ending:** No natural termination, including after a fall Time limit: 1,000 steps, about 50 simulated seconds

**Learning target:** A coordinated gait that maintains speed Time-limit truncation preserves value bootstrapping

Sources:

- https://gymnasium.farama.org/environments/mujoco/half_cheetah/
- https://github.com/MedvAx-AI/iql-halfcheetah/blob/c372ec0423b81839956ec00dd7338700db8d3737/src/iql_project/iql.py

## Slide 6: Offline data and evaluation protocol

Suggested speaker: Ruslan Nasibullin. Time: 40 seconds.

**Dataset:** Minari mujoco/halfcheetah/medium-v0 1,000 episodes and 1,000,000 transitions

**Training budget:** 500,000 offline updates per training seed Seeds 0, 1 and 2, batch size 256

**Evaluation:** Intermediate checkpoints, 10 episodes per run Raw return, uniform random baseline and full rollouts

**Reproducibility:** Dataset-matched environment, MuJoCo 3.2.3 Separate checkpoint selection and final test episodes

Sources:

- https://minari.farama.org/datasets/mujoco/halfcheetah/medium-v0/
- https://github.com/MedvAx-AI/iql-halfcheetah/blob/c372ec0423b81839956ec00dd7338700db8d3737/docs/POLICY_CONTINUATION.md

## Slide 7: Why IQL and HalfCheetah fit

Suggested speaker: Telman Nuruzov. Time: 50 seconds.

**Continuous control:** The policy learns bounded joint torques directly

**Available offline data:** A fixed locomotion dataset supports IQL training without collecting new training rollouts

**Learning opportunity:** Advantage weights emphasize better recorded actions Improvement over the behavior policy is a hypothesis

**Practical environment:** Standard benchmark with measurable returns and visible locomotion behavior

Sources:

- https://arxiv.org/abs/2110.06169
- https://minari.farama.org/datasets/mujoco/halfcheetah/medium-v0/
- https://gymnasium.farama.org/environments/mujoco/half_cheetah/

## Slide 8: Challenges and final-project plan

Suggested speaker: Aleksandr Medvedev. Time: 50 seconds.

**Coverage and stability:** Limited data coverage and seed-dependent failures Multiple seeds, separate final tests and full videos

**Reliable comparisons:** Physics versions and checkpoint selection affect results Pinned dependencies and documented evaluation seeds

**Team ownership:** Aleksandr: integration and reproducibility Nikita: environment and dataset; Leo: IQL Ruslan: evaluation; Telman: notebook and presentation

**Final deliverables:** Reproducible Colab, checkpoints, evaluation plots and a complete policy demonstration

Sources:

- https://github.com/MedvAx-AI/iql-halfcheetah
- https://github.com/MedvAx-AI/iql-halfcheetah/blob/c372ec0423b81839956ec00dd7338700db8d3737/docs/POLICY_CONTINUATION.md
