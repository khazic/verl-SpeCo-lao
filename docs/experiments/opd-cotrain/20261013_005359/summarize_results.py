"""Recompute the 200-step comparison from the accompanying console logs."""

import hashlib
import json
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
METRICS = (
    "drafter/spec_decode/mean_acceptance_length",
    "critic/rewards/mean",
    "perf/time_per_step",
    "timing_s/gen",
    "timing_s/drafter",
    "actor/distillation/abs_loss",
    "actor/pg_loss",
)


def summarize(filename):
    samples = {}
    text = re.sub(r"\x1b\[[0-9;]*m", "", (ROOT / filename).read_text(encoding="utf-8"))
    for line in text.splitlines():
        step = re.search(r"\bstep:(\d+)\s+-", line)
        if step is None or not 1 <= int(step[1]) <= 200:
            continue
        values = {}
        for metric in METRICS:
            match = re.search(re.escape(metric) + r":(-?[\d.eE+]+)", line)
            if match:
                values[metric] = float(match[1])
        if values:
            samples[int(step[1])] = values
    assert set(samples) == set(range(1, 201)), f"Incomplete steps: {filename}"
    return {
        metric: {
            "mean": mean(values) if values else None,
            "count": len(values),
        }
        for metric in METRICS
        for values in [[row[metric] for row in samples.values() if metric in row]]
    }


if __name__ == "__main__":
    results = {
        "fixed_drafter": summarize("run_opd_cotrain_dspark_vllm_fsdp2_no_training.log.txt"),
        "co_train": summarize("run_opd_cotrain_dspark_vllm_fsdp2.log.txt"),
    }
    (ROOT / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    checksums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(ROOT.iterdir())
        if path.suffix in {".png", ".sh"} or path.name.endswith(".log.txt")
    }
    (ROOT / "sha256.json").write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
