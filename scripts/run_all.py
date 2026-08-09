"""Reproduce every number, figure and transcript in this repository.

    python scripts/run_all.py            # ~6 minutes, no network needed
    python scripts/run_all.py --fetch    # also re-download the raw trace

Each stage's real stdout/stderr is written to evidence/<n>-<stage>.txt with the
command echoed above it, so anything quoted in the report or the write-up can be
traced back to a transcript rather than to a claim.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

STAGES = [
    ("01-explore", [
        ["scripts/explore.py", "--dataset", "alibaba"],
        ["scripts/explore.py", "--dataset", "synthetic"],
    ]),
    ("02-forecast", [
        ["scripts/forecast_model.py", "--dataset", "alibaba", "--current-replicas", "20"],
        ["scripts/forecast_model.py", "--dataset", "synthetic", "--current-replicas", "20"],
    ]),
    ("03-rolling-eval", [
        ["scripts/rolling_eval.py", "--dataset", "alibaba"],
        ["scripts/rolling_eval.py", "--dataset", "synthetic"],
    ]),
    ("04-reproducibility", [
        ["scripts/reproducibility.py", "--runs", "12"],
    ]),
    ("05-simulation", [
        ["scripts/simulate.py", "--days", "7"],
    ]),
    ("06-sweeps", [
        ["scripts/sweep.py"],
    ]),
    ("07-exporter", [
        ["scripts/exporter.py", "--once", "--port", "8031"],
    ]),
    ("08-tests", [
        ["-m", "pytest", "-q"],
    ]),
]


def run(stage: str, commands: list[list[str]]) -> bool:
    out = ROOT / "evidence" / f"{stage}.txt"
    out.parent.mkdir(exist_ok=True)
    ok = True
    with out.open("w", encoding="utf-8") as fh:
        for cmd in commands:
            line = "python " + " ".join(cmd)
            print(f"  $ {line}")
            fh.write(f"$ {line}\n")
            fh.flush()
            t0 = time.time()
            proc = subprocess.run([PY, *cmd], cwd=ROOT, capture_output=True, text=True)
            body = proc.stdout + proc.stderr
            # Prophet's plotly warning is emitted at import and says nothing
            # about this run; everything else goes through untouched.
            body = "\n".join(l for l in body.splitlines()
                             if "Importing plotly failed" not in l)
            fh.write(body.rstrip() + "\n")
            fh.write(f"[exit {proc.returncode}, {time.time()-t0:.1f}s]\n\n")
            if proc.returncode != 0:
                ok = False
                print(f"    !! exit {proc.returncode}")
    print(f"  -> {out.relative_to(ROOT)}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="re-download the raw Alibaba rows (needs network)")
    ap.add_argument("--only", nargs="*", help="run only these stage names")
    args = ap.parse_args()

    stages = list(STAGES)
    if args.fetch:
        stages.insert(0, ("00-data", [
            ["scripts/fetch_alibaba.py", "--machines", "24"],
            ["scripts/prepare_data.py"],
        ]))
    if args.only:
        stages = [s for s in stages if s[0] in args.only]

    started = time.time()
    failed = []
    for name, cmds in stages:
        print(f"\n== {name} ==")
        if not run(name, cmds):
            failed.append(name)

    print(f"\ndone in {time.time()-started:.0f}s")
    if failed:
        print("FAILED stages:", ", ".join(failed))
        raise SystemExit(1)
    print("all stages exited 0")


if __name__ == "__main__":
    main()
