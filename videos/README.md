# Videos

Person 4 supplies `videos/<run_id>/demo_seed_<seed>.mp4` from the trained agent.
Use `iql-project evaluate --checkpoint ... --run-dir ... --video`, or the
three-seed experiment script described in [the evaluation guide](../docs/EVALUATION.md).
The preselected video seed is 10000, with the same deterministic policy and
checkpoint preprocessing used for metrics. Existing video filenames are protected.
Use Releases/team storage for large files, and link the URL and SHA-256 in the
manifest and final notebook. Generated local videos are excluded from Git;
their existence alone does not mean they have been published to shared storage.

## Published 100k-update demos

All three final-checkpoint episodes use evaluation seed **10000**, last **50 seconds**,
and encode at **480×480, 20 fps**. They show falls or stalls; robust running has not
been demonstrated. The whole episode is retained, including the failure.

- [Training seed 0](https://github.com/MedvAx-AI/iql-halfcheetah/releases/download/person4-evaluation-100k/person4_100k_seed_0_visible_floor_demo_seed_10000.mp4):
  SHA-256 `42a63d4fe617edb1cd446d9bf40701f94e4e03a027b179b9321278b8476064fe`.
- [Training seed 1](https://github.com/MedvAx-AI/iql-halfcheetah/releases/download/person4-evaluation-100k/person4_100k_seed_1_visible_floor_demo_seed_10000.mp4):
  SHA-256 `f34883f3411e02ffac3e44d2389636bc8719d6ef01812dab37d1f4d51bac63c0`.
- [Training seed 2](https://github.com/MedvAx-AI/iql-halfcheetah/releases/download/person4-evaluation-100k/person4_100k_seed_2_visible_floor_demo_seed_10000.mp4):
  SHA-256 `d4c080a30e5ced91d70b14f28e0ad08faa630343b6c6d62d0a8f074f36518abb`.

The [release](https://github.com/MedvAx-AI/iql-halfcheetah/releases/tag/person4-evaluation-100k)
also contains the original videos and an archive with checkpoints/logs. The
[publication manifest](../docs/evidence/person-4/publication.json) records URLs,
sizes, SHA-256, seeds, checkpoint identity and anonymous download checks.

The default MuJoCo floor is physically infinite but drawn only within ±40 m.
Video recording now draws that plane as infinite too, so the tracking camera shows
the ground throughout. This changes drawing only: the repeated episode returns and
lengths match the original CSV exactly. See the [measured demo diagnostics](../docs/evidence/person-4/demo_diagnostics.json).
