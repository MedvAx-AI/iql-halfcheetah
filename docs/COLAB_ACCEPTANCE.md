# Colab execution and final acceptance

[Open the project notebook](https://colab.research.google.com/github/MedvAx-AI/iql-halfcheetah/blob/main/notebooks/iql_halfcheetah.ipynb)
and select **Run all** on a CPU runtime. No paid GPU, Drive mount, manual repository
checkout or package installation is required. The setup cell creates an isolated
Python 3.11 kernel from the frozen lock, so Colab's preloaded libraries and host
Python version do not replace the experiment's numerical stack.

Project cells use `%%project`; their variables persist in the project kernel, and
text/plots/video are forwarded to the Colab interface. The setup cell runs in Colab's
host kernel. Re-running setup closes the old project kernel and starts a new one.
For local use, launch Jupyter through the locked environment as shown in the README.

Run all performs:

1. Package/dependency checks and a fresh dataset download if not cached.
2. Real environment reset, 1,000,000-transition loader validation and numerical example.
3. A fresh 1,000-update training run with saved checkpoints and sequential metrics.
4. Checksum-verified retrieval of three published 100k-update final checkpoints.
5. Two replays per checkpoint, each using ten IQL and ten random episodes; all
   60 episode records must match their repeat exactly on this platform.
6. Replay of a checksum-verified continued checkpoint, an exact repeat of its
   20 IQL/random validation records, and ten additional IQL/random episodes.
   Its scores are shown separately from the original 100k experiment.
7. A fresh 50-second continued-policy MP4 with first/middle/end frame decoding,
   plus its evaluation plot. The original replay also retains its own MP4.
8. The full test suite with real-data training tests enabled and dataset checksum check.
9. An acceptance JSON and an evidence ZIP under the printed `results/notebook-replay-*`
   directory. Runtime files are temporary; download the ZIP through Colab's Files pane
   before disconnecting if you need your own run's logs, raw CSVs, plots and video.

Failures stop execution. There are no successful env/data SKIP fallbacks. Run all
does not retrain the original three 100k schedules or the default 500k schedule;
those measured experiments are preserved in the published release.

The final cell prints the actual checkout SHA and numerical runtime. To run an
immutable revision, set `IQL_GIT_REF` to its commit before setup. Otherwise a fresh
runtime clones current main. Colab may ask you to confirm running a GitHub notebook;
review the source, then use the normal Run anyway control.

See [reproducibility scope](REPRODUCIBILITY.md) for separately labeled platform
returns and the simulator-only investigation.

## Verified fresh Colab Run all — 9 October 2026

The committed notebook completed through Colab's **Run all** control in a new,
default CPU session, without changing cells, installing packages manually,
setting a revision override, mounting Drive or downgrading the host runtime.

- Actual checkout: `e710dcb33621861c9c1d1e14735a3fd7a0beb8dd` (merged PR #13).
- Colab host: Python 3.13.16; project kernel: Python 3.11.13.
- Platform: Linux 6.6.122+, x86_64, glibc 2.39; frozen CPU dependency stack.
- All **11 code cells** completed. No environment/dataset skip fallbacks.
- Fresh download: **1,000,000 transitions**; dataset SHA-256 matched the published data.
- Fresh training: **1,000 sequential updates**, with a saved checkpoint and log hashes.
- Three published 100k checkpoints: archive and individual checkpoint hashes matched;
  all **60 individual IQL/random episode records** matched their second replay exactly.
- Current-platform aggregate: **2187.95 ± 480.18**, separately labeled from the
  original macOS **2846.41 ± 786.80** result.
- New MP4 recorded and decoded: 480×480, 50 seconds, 20 fps, first/middle/end frames.
- Full real-data test suite: **92 passed in 42.29 seconds**, no warnings in this run.
- Final cell reported `RUN ALL ACCEPTANCE PASSED`.

[Raw acceptance JSON and all 60 episode records](evidence/colab/2026-10-09-e710dcb.json)
and [the visible Colab result](evidence/colab/2026-10-09-run-all.jpg) are retained.
See [evidence provenance](evidence/colab/README.md) for the difference between
frontend evidence and the separate CI artifact bundle.

The original three 100k training runs were replayed, not retrained. The default
500k schedule and robust locomotion remain explicitly outside this historical acceptance
claim; those original fixed-seed demos show falling/stalling. The implementation and
documented course deliverables pass final integration acceptance within that scope.

The plot correction at merged commit `f1ba5f1fda010165f9715a4a7e746f143ac36657`
also completed all 11 original cells in a fresh Colab session: **94 tests passed
in 41.90 seconds**, all 60 records repeated exactly and the video decoded.
See its [raw acceptance JSON](evidence/colab/2026-10-09-f1ba5f1.json).
The later continuation experiment and its separate performance limits are
described in [POLICY_CONTINUATION.md](POLICY_CONTINUATION.md).
