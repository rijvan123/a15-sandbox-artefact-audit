#!/usr/bin/env python
"""How many S10 worker processes, and how many threads each, train fastest on this machine (7 Nov 2026).

Reads no study data. Each worker trains a Transformer encoder and a BiLSTM (configurations of the registered search,
L = 256, batch 32) on random tokens for a fixed time, and the script reports the total training throughput of every
combination. The S10 evaluation is many independent training runs, so throughput across workers is what decides
the wall time. It changes speed only, not results.

    python bench_parallel.py                      # about 6 minutes; prints a table and the fastest combination
    python bench_parallel.py --combos 1x8,4x3 --seconds 30
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CFG = {"transformer": {"layers": 3, "ff": 512, "dropout": 0.1}, "bilstm": {"layers": 1, "hidden": 64, "dropout": 0.1}}


def _worker(kind: str, threads: int, seconds: float, warm: float, q) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "A15_THREADS"):
        os.environ[var] = str(threads)
    import torch
    torch.set_num_threads(threads)
    from a15.models import BUILDERS
    torch.manual_seed(0)
    V, K, L, B = 4000, 9, 256, 32
    model = BUILDERS[kind](V, K, **CFG[kind])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids, betas = torch.randint(4, V, (B, L)), torch.randint(0, 8, (B, L))
    y = torch.randint(0, K, (B,))
    n, t_end, t_start = 0, time.perf_counter() + warm + seconds, time.perf_counter() + warm
    while time.perf_counter() < t_end:
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(ids, betas), y).backward()
        opt.step()
        if time.perf_counter() >= t_start:
            n += B
    q.put(n / seconds)


def combo(kind: str, workers: int, threads: int, seconds: float, warm: float) -> float:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_worker, args=(kind, threads, seconds, warm, q)) for _ in range(workers)]
    for p in ps:
        p.start()
    rates = [q.get() for _ in ps]
    for p in ps:
        p.join()
    return float(sum(rates))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", default="1x8,2x4,3x4,4x3,6x2",
                    help="workers x threads; keep workers x threads near the number of logical CPUs")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--warm", type=float, default=8.0)
    a = ap.parse_args()
    combos = [tuple(int(v) for v in c.split("x")) for c in a.combos.split(",")]
    rows = []
    print(f"logical CPUs: {os.cpu_count()}; {a.seconds:.0f} s per measurement after {a.warm:.0f} s warm-up", flush=True)
    for w, t in combos:
        r = {"workers": w, "threads": t}
        for kind in ("transformer", "bilstm"):
            r[kind] = combo(kind, w, t, a.seconds, a.warm)
        rows.append(r)
        print(f"{w} worker(s) x {t} thread(s): Transformer {r['transformer']:7.1f} traces/s, "
              f"BiLSTM {r['bilstm']:7.1f} traces/s", flush=True)
    base = rows[0]
    for r in rows:                     # S10 time is about 2/3 Transformer and 1/3 BiLSTM (K6 projection)
        r["speedup_vs_first"] = 1 / ((2 / 3) * base["transformer"] / r["transformer"]
                                     + (1 / 3) * base["bilstm"] / r["bilstm"])
    best = max(rows, key=lambda r: r["speedup_vs_first"])
    print(json.dumps({"rows": rows, "fastest": {"workers": best["workers"], "threads": best["threads"],
                                                "speedup_vs_first": round(best["speedup_vs_first"], 2)}}, indent=1))
    print(f"\nUse: run_s10.py ... evaluate --workers {best['workers']} --threads {best['threads']}")


if __name__ == "__main__":
    main()
