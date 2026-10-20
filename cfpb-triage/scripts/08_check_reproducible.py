"""Entry point: 08_check_reproducible (Sayon). Re-runs 04_run_baseline.py and checks
that every metric in results/tables/baseline_runs.csv comes out identical.

Timings are allowed to differ; metrics are not. Also records the environment
(python + package versions, data file hashes) in results/logs/repro_check.json.
"""
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version

import pandas as pd

from cfpb_triage.data.load import resolve

METRICS = ["macro_f1", "weighted_f1", "accuracy"]
PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "pyyaml"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def main() -> None:
    runs_path = resolve("results/tables/baseline_runs.csv")
    if not runs_path.exists():
        sys.exit("[08] run scripts/04_run_baseline.py first")
    before = pd.read_csv(runs_path)
    subprocess.run([sys.executable, str(resolve("scripts/04_run_baseline.py"))], check=True,
                   stdout=subprocess.DEVNULL)
    after = pd.read_csv(runs_path)

    same = before[["model", "seed"] + METRICS].equals(after[["model", "seed"] + METRICS])
    max_diff = float((before[METRICS] - after[METRICS]).abs().max().max())
    report = {
        "identical_metrics": bool(same),
        "max_abs_metric_diff": max_diff,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in PACKAGES},
        "data_sha256_16": {name: sha256(resolve(f"data/processed/splits/{name}.csv"))
                           for name in ("train", "val", "test")},
    }
    resolve("results/logs").mkdir(parents=True, exist_ok=True)
    resolve("results/logs/repro_check.json").write_text(json.dumps(report, indent=2),
                                                        encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not same:
        sys.exit(f"[08] FAIL: metrics changed between runs (max diff {max_diff})")
    print("[08] PASS: two runs gave identical metrics")


if __name__ == "__main__":
    main()
