#!/usr/bin/env python3
"""Verify this release: MANIFEST.sha256, the frozen code and the measured results file.

    python3 verify_release.py
"""
import hashlib
import sys
from pathlib import Path

RESULTS_SHA256 = "b2e80138bcbc1217162c046110a9a62880ce7361aee7c5c3e6a2e0c0c3e7b692"   # ledger entry of the results


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    here, problems = Path(__file__).resolve().parent, 0
    man = here / "MANIFEST.sha256"
    if man.exists():
        lines = [l for l in man.read_text().splitlines() if l.strip()]
        bad = [n for h, n in (l.split(None, 1) for l in lines) if not (here / n).exists() or sha(here / n) != h]
        print(f"MANIFEST.sha256: {len(lines) - len(bad)} of {len(lines)} files match")
        problems += len(bad)
        for n in bad[:20]:
            print("  differs or missing:", n)
    else:
        print("MANIFEST.sha256 not found (run collect_release.py first)")
    eng, sums = here / "s10_engine", here / "provenance" / "SHA256SUMS_s10_engine.txt"
    rows = [l.split(None, 1) for l in sums.read_text().splitlines() if l.strip()]
    ok = sum(1 for h, n in rows if (eng / n.strip()).exists() and sha(eng / n.strip()) == h)
    print(f"frozen code: {ok} of {len(rows)} files match provenance/SHA256SUMS_s10_engine.txt")
    problems += len(rows) - ok
    res = here / "results" / "results.json"
    match = res.exists() and sha(res) == RESULTS_SHA256
    print(f"results/results.json: {'matches' if match else 'DOES NOT MATCH'} the ledger digest {RESULTS_SHA256[:16]}...")
    problems += 0 if match else 1
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
