"""Experiment log (Sayon). Every M5 script wraps its work in `logged_run(...)`.

Appends one JSON line per run to m5_deliverables/results/logs/experiment_log.jsonl
(who/when, script, arguments, git commit + dirty flag, package versions, seed, wall time,
outputs, and a free-text note) and can render the whole history as a markdown table:

    python m5_deliverables/code/m5_runlog.py --render
    python m5_deliverables/code/m5_runlog.py --note "kNN k=51 gave no gain over k=25 (negative result)"

Negative and inconclusive runs must be logged too (spec section 6): use --note or the
`note=` argument. Never delete lines from the .jsonl file.
"""
from __future__ import annotations

import argparse
import contextlib
import getpass
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

LOGS = Path(__file__).resolve().parents[1] / "results" / "logs"
JSONL = LOGS / "experiment_log.jsonl"
MD = LOGS / "experiment_log.md"


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).parent, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _versions() -> dict:
    import numpy, pandas, scipy, sklearn
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__}


def append(entry: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


@contextlib.contextmanager
def logged_run(script: str, args: dict | None = None, seed: int = 42, note: str = ""):
    """with logged_run("m5_tune", vars(args)) as run:  ...  run["outputs"] = [...]"""
    run = {"outputs": [], "note": note}
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield run
    except BaseException:
        status = "FAILED"
        raise
    finally:
        append({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "user": getpass.getuser(), "script": script, "args": args or {},
                "seed": seed, "status": status, "wall_s": round(time.perf_counter() - t0, 1),
                "git_commit": _git("rev-parse", "--short", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain")),
                "versions": _versions(), "outputs": run["outputs"], "note": run["note"]})


def render() -> str:
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")] if JSONL.exists() else []
    out = ["# M5 experiment log", "",
           "Generated from `experiment_log.jsonl` (append-only; failed and negative runs kept).", "",
           "| # | UTC time | script | status | wall (s) | commit | args | note |",
           "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        commit = r["git_commit"] + ("*" if r.get("git_dirty") else "")
        args = ", ".join(f"{k}={v}" for k, v in r["args"].items())
        out.append(f"| {i} | {r['time_utc']} | {r['script']} | {r['status']} | {r['wall_s']} | "
                   f"{commit} | {args} | {r['note']} |")
    out += ["", "`*` = uncommitted changes when the run started."]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="write experiment_log.md")
    ap.add_argument("--note", help="append a manual entry (decision, negative result, deviation)")
    a = ap.parse_args()
    if a.note:
        with logged_run("manual_note", note=a.note):
            pass
    if a.render or a.note:
        LOGS.mkdir(parents=True, exist_ok=True)
        MD.write_text(render(), encoding="utf-8")
        print("wrote", MD)
