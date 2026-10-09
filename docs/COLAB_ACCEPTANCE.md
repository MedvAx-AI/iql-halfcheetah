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
6. A fresh 50-second MP4 with first/middle/end frame decoding, plus the evaluation plot.
7. The full test suite with real-data training tests enabled and dataset checksum check.
8. An acceptance JSON and an evidence ZIP under the printed `results/notebook-replay-*`
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
returns and the simulator-only investigation. Fresh-run verification evidence will
be appended here after the committed notebook has completed in Google Colab.
