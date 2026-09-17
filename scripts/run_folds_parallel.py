"""Run an OOF experiment's folds concurrently instead of one after another.

WHY THIS EXISTS. Every gate in this project fits three folds sequentially, and each fold fits a
LightGBM and a CatBoost one after the other. Measured on the biggest fold, `fit()` is ~90% of wall
clock and frame building ~10%, so the whole cost is GBM training -- and GBM thread-scaling is
strongly sublinear. One fold on 12 threads does not go three times faster than three folds on 4
threads each; it goes barely faster at all past ~6 threads. Running the folds concurrently is
therefore close to free wall-clock.

WHY SUBPROCESSES AND NOT THREADS OR A PROCESS POOL. The panel (`pipeline4.TR`, 1.2M rows) is a
module-level global built at import. Passing it to pool workers means pickling it three times;
threads would contend on the GIL for the pandas work and, worse, LightGBM and CatBoost each manage
their own thread pools. A plain subprocess per fold sidesteps both: each process imports the panel
itself (in parallel with the others) and owns its own BLAS/OMP thread budget.

THREAD BUDGET. Each child is capped via OMP_NUM_THREADS / MKL_NUM_THREADS so three children do not
collectively oversubscribe the box. Oversubscription is worse than sequential -- the threads fight
over cache and the run slows down.

DETERMINISM CAVEAT, worth stating. LightGBM's histogram accumulation order depends on the thread
count, so the same seed at 4 threads and at 12 threads can differ in the last few decimal places.
That is far below this project's 0.015 seed spread and it is applied identically to every arm, so
paired comparisons are unaffected -- but a number produced here will not be bit-identical to one
produced sequentially. Do not mix cached arms across thread budgets without re-running both.

Usage:
    python scripts/run_folds_parallel.py src/experiments/oof_coldshape.py [--threads 4]

The target script must accept `--fold N` and be safe to run once per fold.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 8) // 3))
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()

    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = str(a.threads)

    print(f"launching {a.folds} folds x {a.threads} threads "
          f"({a.folds * a.threads} of {os.cpu_count()} logical cores)", flush=True)
    t0 = time.time()
    procs, logs = [], []
    for i in range(a.folds):
        log = ROOT / "outputs" / f"_parallel_fold{i}.log"
        fh = open(log, "w", encoding="utf-8")
        cmd = [sys.executable, a.script, "--fold", str(i), *a.extra]
        procs.append((i, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh,
                                          stderr=subprocess.STDOUT), fh))
        logs.append(log)
        print(f"  fold {i} -> pid {procs[-1][1].pid}  log {log.name}", flush=True)

    rc = 0
    for i, p, fh in procs:
        r = p.wait()
        fh.close()
        tail = (ROOT / "outputs" / f"_parallel_fold{i}.log").read_text(
            encoding="utf-8", errors="replace").strip().split("\n")[-1:]
        print(f"  fold {i} exit {r}  | {tail[0] if tail else ''}", flush=True)
        rc = rc or r
    print(f"all folds done in {time.time()-t0:.0f}s (exit {rc})", flush=True)
    if rc:
        print("  a fold FAILED -- check outputs/_parallel_fold*.log", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
