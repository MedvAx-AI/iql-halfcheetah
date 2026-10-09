# Fresh Colab frontend evidence

The plot correction was independently rerun in a fresh Colab session at merged
commit `f1ba5f1fda010165f9715a4a7e746f143ac36657`. All 11 original code cells
completed with **94 tests passing in 41.90 seconds**, real data, fresh 1,000-update
training, all 60 exact-repeat records and a decoded video.
[`2026-10-09-f1ba5f1.json`](2026-10-09-f1ba5f1.json) retains that actual visible
acceptance output. Temporary experimental audit cells were added only after
this unchanged notebook run completed; their outputs are retained separately
in [continuation evidence](../continuation/).

Recorded 9 October 2026 at merged source commit
`e710dcb33621861c9c1d1e14735a3fd7a0beb8dd`.

`2026-10-09-e710dcb.json` is the JSON printed by the committed notebook's final
acceptance cell, read from its visible Colab output. It retains the observed host
and project runtimes, original dataset/checkpoint hashes, fresh training hashes,
test output, separately labeled published/current-platform summaries and all 60
raw episode records. It is not a simulated or manually computed acceptance result.

`2026-10-09-run-all.jpg` shows the actual 92-test success and final acceptance
message in Colab. The narrow screenshot reflects the app's browser pane.

`2026-10-09-final-checkpoint-returns.png` re-renders the same 30 IQL episode
records from `2026-10-09-e710dcb.json` with separate training-seed positions,
individual episode dots and mean ± population episode standard deviation. This
is a presentation correction of the retained Colab data, not a new evaluation
or evidence of policy improvement. The random reference mean uses the same
evaluation seed set.

The source notebook was opened in a new Colab tab/session, and **Run all** was
selected. Only Colab's standard GitHub-notebook Run anyway confirmation was used;
no source-cell modifications or manual preparation were needed.

The runtime created a full evidence ZIP, including raw evaluations, plots, fresh
video, logs and the fresh 1,000-update checkpoint. Its files remain temporary in
Colab. The final release also provides a separately labeled **Linux CI** evidence
bundle downloaded from the successful merged-commit workflow
[37899522313](https://github.com/MedvAx-AI/iql-halfcheetah/actions/runs/37899522313).
That CI bundle is not represented as this Colab session's ZIP. Its SHA-256 is
`63a358367b10bedc6af63c9b51591abcefa8d37eeb2423a0d3e329f1cff6b421`.

See [acceptance scope](../../COLAB_ACCEPTANCE.md) and
[reproducibility investigation](../../REPRODUCIBILITY.md). No full retraining of
the three published 100k schedules, 500k experiments or robust locomotion is claimed.
