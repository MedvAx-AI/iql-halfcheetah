# Videos

Person 4 supplies `videos/<run_id>/demo_seed_<seed>.mp4` from the trained agent.
Use `iql-project evaluate --checkpoint ... --run-dir ... --video`, or the
three-seed experiment script described in [the evaluation guide](../docs/EVALUATION.md).
The preselected video seed is 10000, with the same deterministic policy and
checkpoint preprocessing used for metrics. Existing video filenames are protected.
Use Releases/team storage for large files, and link the URL and SHA-256 in the
manifest and final notebook. Generated local videos are excluded from Git;
their existence alone does not mean they have been published to shared storage.
