"""Thread-count check for the K6 input on the analysis machine (amendment 4, 5 November 2026).

models.py runs PyTorch on every logical CPU unless A15_THREADS is set. On machines with many logical CPUs,
hyper-threads or efficiency cores, small sequence models can then run several times slower than on a few threads,
because every operation waits for the slowest thread. This script times one forward-backward pass per trace at the
sequence cap L for each learner (the configuration the dry run timed before amendment 4: first draw of seed 0, 16
traces, best of 3 passes) at several thread counts, prints the projected hours of the CPU plan for each, and names the
fastest. It reads no data, writes no file and changes no result: the thread count only changes how fast the same
computation runs.

    python bench_threads.py                  # thread counts 1, 2, 4, ... up to the number of logical CPUs
    python bench_threads.py --threads 2,4,8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                                       # noqa: E402

from a15 import config as C, gates, models as M                     # noqa: E402

LEARNERS = ("transformer", "bilstm", "cnn")
_RUNS = C.N_SEARCH + len(C.DROPOUT_GRID) + 3 * len(C.SEEDS)       # runs per cell, as in run_s9.py dryrun
RUNS_PER_CELL = {"transformer": _RUNS, "bilstm": 2 * len(C.SEEDS), "cnn": _RUNS}   # CPU plan (amendments 3, 5)
N_TRAIN_4NOV = {"I1": 2302.6666666666665, "I2": 5241.0}            # full-design training sizes, dry run of 4 Nov 2026


def plan_hours(sec: dict) -> dict:
    h = {k: gates.k6_projection({k: sec[k]}, N_TRAIN_4NOV, origins={"I1": 3, "I2": 1},
                                runs_per_cell=RUNS_PER_CELL[k])["total_hours"] for k in sec}
    return {**h, "total": sum(h.values())}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", default="", help="comma-separated thread counts (default: 1, 2, 4, ... all)")
    ap.add_argument("--L", type=int, default=C.L_CAP)
    ap.add_argument("--V", type=int, default=10_400, help="vocabulary size (lexicon of 4 Nov: 10,398 token types)")
    ap.add_argument("--K", type=int, default=9, help="number of families")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    n_cpu = os.cpu_count() or 1
    ts = ([int(x) for x in a.threads.split(",") if x.strip()] if a.threads else
          sorted({t for t in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64) if t <= n_cpu} | {n_cpu}))
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    print(f"logical CPUs {n_cpu}; A15_THREADS={os.environ.get('A15_THREADS', '(not set)')}; load average {load}; "
          f"L {a.L}, V {a.V}, K {a.K}; torch {torch.__version__}", flush=True)
    print(f"{'threads':>7} {'transformer':>12} {'bilstm':>9} {'cnn':>9}   seconds per trace; CPU plan, hours", flush=True)
    rows = []
    t0 = time.perf_counter()
    for t in ts:
        torch.set_num_threads(t)
        sec = {k: M.time_per_trace(k, a.V, a.K, a.L, M.sample_configs(k, 1, 0)[0], n=a.n, reps=a.reps)
               for k in LEARNERS}
        h = plan_hours(sec)
        rows.append({"threads": t, "sec": sec, "hours": h})
        print(f"{t:>7} {sec['transformer']:>12.4f} {sec['bilstm']:>9.4f} {sec['cnn']:>9.4f}   {h['total']:,.0f} h",
              flush=True)
    best = min(rows, key=lambda r: r["hours"]["total"])
    print(f"\nfastest: {best['threads']} threads, CPU plan about {best['hours']['total']:,.0f} h at this machine's "
          f"current speed ({time.perf_counter() - t0:,.0f} s for this check).")
    print(f"To use it, run once:  echo 'export A15_THREADS={best['threads']}' >> ~/a15/a15env.sh && "
          f"source ~/a15/a15env.sh")
    return {"rows": rows, "best": best}


if __name__ == "__main__":
    main()
