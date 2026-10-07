# Results

Person 4 writes run-scoped raw metrics, summaries and plots as specified in
`docs/INTERFACES.md`. Generated artifacts are ignored. Publish reviewed summaries
and artifact links only after real runs, with commit/config/seed provenance.

See [the evaluation guide](../docs/EVALUATION.md) for the checkpoint CLI, fixed
evaluation seeds, random baseline, plotting and aggregation commands. Within-run
episode variation and across-run variation of training-seed means are reported
separately. `summary.json` marks incomplete seed/episode comparisons explicitly.
