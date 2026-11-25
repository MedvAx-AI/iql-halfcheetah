# IQL on HalfCheetah-v5 | Final defense

Use [the project notebook](../../notebooks/iql_halfcheetah.ipynb) as the main presentation. Open the MP4 locally before the session and have §§5–10 ready in Colab.

| Time | What to show |
|---|---|
| 0:00–0:56 | Play `IQL_Training_Progress_Demo.mp4`. Explain that the first three excerpts use training seed 1 and the same initial-state seed, 10002. The final excerpt shows a 500k failure from a different run. |
| 0:56–5:26 | Spend 4½ minutes on the training code and settings in notebook §8. Follow the sequence below. |
| 5:26–17:00 | Present the algorithm from notebook §5, then work through the numerical update in §6. Give most of this time to the example and how it maps to the implementation. |
| 17:00–22:00 | Show the learning curves and measured returns in §§9–10. Explain why one good episode and one bad episode are illustrations and why aggregate and held-out evaluation matter. |
| 22:00–30:00 or 35:00 | Leave the remaining group time for the instructor’s questions. |

## Training code in 4½ minutes

1. **1 minute.** Open [the config](../../configs/halfcheetah.toml). HalfCheetah-v5 receives six bounded actions and returns 17 observation values. Training samples 256 transitions at a time from one million recorded Minari transitions.
2. **1½ minutes.** In [the training loop](../../src/iql_project/train.py), find the minibatch sample, `agent.update(batch)`, logging and checkpoints. Explain that 100k, 300k and 500k are optimizer update counts. Evaluation runs separate 1,000-step simulator episodes. `Run all` runs a 1,000-update training smoke check and replays saved policies.
3. **2 minutes.** In [the learner](../../src/iql_project/iql.py), find `IQLAgent.update`. Walk through V, the twin Q critics, and the policy, followed by the target-critic soft update. Point out expectile 0.7, discount 0.99, inverse temperature 3, weight cap 100, Adam learning rate 0.0003 and target update coefficient 0.005.

## Work the example live

Start with the smaller frozen target Q at 8 and the current V at 5. The gap is 3. The positive residual gets weight 0.7, so the expectile loss is 6.3 and its gradient with respect to V is −4.2. A teaching step of size 0.1 moves V to **5.42** and lowers the loss to about **4.659**. In training, the same loss updates the value network through Adam.

Next, with reward 1 and next V equal to 6, the critic target is **6.94** on a continuing transition and **1** on a true terminal transition. HalfCheetah's time limit is a truncation. Finally, hold the original Q and V fixed for hand calculation. The action has advantage 3, so its policy weight is `min(exp(3 × 3), 100) = 100`. The implementation computes the actor's advantage after the value update.

## Demo cues

| Video time | What to call out |
|---|---|
| 0:04–0:16 | 100k, seed 1. Early stall. |
| 0:16–0:28 | 300k, seed 1. Runs for longer, then stalls. The on-screen timer shows where the excerpt starts. |
| 0:28–0:40 | 500k, seed 1. Keeps moving through the final seconds. |
| 0:40–0:52 | 500k, seed 0. A separately selected episode still stalls. |

The footage uses the saved checkpoints and MuJoCo 3.2.3. All excerpts play at real speed. See the accompanying provenance for checkpoint hashes, exact episode windows and full-episode returns. The optional 50-second source clips are for inspection. Play only the 56-second montage during the defense.

The separate ten-episode held-out comparison in [the recorded project results](../POLICY_CONTINUATION.md) gives a mean raw return of about 2,026 at 100k and 15,260 at 500k for training seed 1. The demo episodes were selected to illustrate behaviors. Use the held-out comparison and all three training curves for performance claims.
