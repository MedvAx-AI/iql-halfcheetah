"""Rebuild continuation curves from the retained, observed Colab episode records."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from iql_project.reporting import CSV_FIELDS, _plot, within_run


def main():
    root = Path(__file__).resolve().parent
    audits = {
        step: json.loads((root / f"colab-all-seeds-{step // 1000}k.json").read_text())
        for step in (150000, 200000, 300000, 500000)
    }
    reference = audits[150000]
    provenance = []
    for step, audit in audits.items():
        path = root / f"colab-all-seeds-{step // 1000}k.json"
        provenance.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_commit": audit["source_commit"],
                "platform": audit["platform"],
                "mujoco": audit["mujoco"],
                "torch": audit["torch"],
                "audit_session": audit["audit_session"],
                "checkpoints_sha256": {
                    name: record["checkpoint_sha256"] for name, record in audit["runs"].items()
                },
            }
        )
    for audit in audits.values():
        assert audit["audit_session"] == reference["audit_session"], "Audit sessions differ"
        assert (audit["platform"], audit["mujoco"], audit["torch"]) == (
            reference["platform"],
            reference["mujoco"],
            reference["torch"],
        )
        for seed in range(3):
            assert (
                audit["runs"][f"seed_{seed}_baseline_100k"]
                == reference["runs"][f"seed_{seed}_baseline_100k"]
            ), "Paired baseline changed between measurements"

    for group, start in (("validation", 10000), ("additional", 20000)):
        output = root / group
        output.mkdir(exist_ok=True)
        rows = []
        for seed in range(3):
            stages = {100000: reference["runs"][f"seed_{seed}_baseline_100k"]}
            stages.update(
                {
                    step: audit["runs"][f"seed_{seed}_candidate_{step // 1000}k"]
                    for step, audit in audits.items()
                }
            )
            for step, record in stages.items():
                assert record["step"] == step
                episodes = [
                    item
                    for item in record["episodes"]
                    if start <= item["evaluation_seed"] < start + 10
                ]
                assert sorted(item["evaluation_seed"] for item in episodes) == list(
                    range(start, start + 10)
                )
                assert all(item["episode_length"] == 1000 for item in episodes)
                for index, episode in enumerate(episodes):
                    rows.append(
                        dict(
                            zip(
                                CSV_FIELDS,
                                (
                                    f"seed_{seed}",
                                    step,
                                    seed,
                                    episode["evaluation_seed"],
                                    index,
                                    episode["episode_return"],
                                    episode["episode_length"],
                                    "iql",
                                ),
                                strict=True,
                            )
                        )
                    )
        with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summaries = within_run(rows)
        across = []
        for step in (100000, 150000, 200000, 300000, 500000):
            means = np.asarray(
                [row["return_mean"] for row in summaries if row["checkpoint_step"] == step]
            )
            assert len(means) == 3
            across.append(
                {
                    "checkpoint_step": step,
                    "training_seeds": [0, 1, 2],
                    "mean_of_run_means": float(means.mean()),
                    "std_sample_of_run_means": float(means.std(ddof=1)),
                }
            )
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "source": "Observed Colab audits; reused evaluation sets",
                    "platform": reference["platform"],
                    "mujoco": reference["mujoco"],
                    "torch": reference["torch"],
                    "audit_session": reference["audit_session"],
                    "input_audits": provenance,
                    "within_run": summaries,
                    "across_runs": across,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _plot(output, rows, None)
        print(group, "150 actual episodes; 3 training seeds × 5 checkpoints × 10 episodes")


if __name__ == "__main__":
    main()
