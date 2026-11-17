#!/usr/bin/env python3
"""Complete this repository from the author's work folder before uploading it to GitHub and Zenodo.

    python3 collect_release.py --root /mnt/d/EMBER2024/_A15_S9

Copies the frozen code (s10_engine/) and stops unless every file listed in provenance/SHA256SUMS_s10_engine.txt (and
in any SHA256SUMS_*.txt inside s10_engine/) matches its digest. Then copies the evaluation outputs (WORK/outputs/s10),
the freeze bundle WITHOUT the embargoed token-level lexicon (its digest is written instead), the ledger summary, the
final Data Provenance Log and the change notes, and writes MANIFEST.sha256 for every file in the repository.
Datasets, sealed trace records, caches and the full ledger are never copied.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.pkl", "*.npy", "*.npz", "*.pt", "*.7z",
                              "*.zip", ".venv", "venv", "WORK", "*.log")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(eng: Path, sums: Path) -> list:
    bad = []
    for line in sums.read_text().splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            f = eng / name.strip().lstrip("*")
            if not f.exists() or sha(f) != h:
                bad.append(name.strip())
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="study folder holding s10_engine/ and WORK/")
    a = ap.parse_args()
    root, here = Path(a.root), Path(__file__).resolve().parent
    eng = root / "s10_engine"
    if not eng.is_dir():
        sys.exit(f"no s10_engine folder in {root}")
    lists = [here / "provenance" / "SHA256SUMS_s10_engine.txt"] + sorted(eng.glob("SHA256SUMS_*.txt"))
    for sums in lists:
        bad = check(eng, sums)
        if bad:
            sys.exit(f"stopped: {len(bad)} file(s) differ from {sums.name}: {bad}")
        print(f"verified against {sums.name}")
    dst = here / "s10_engine"
    for old in dst.iterdir():
        if old.name != "README.md":
            shutil.rmtree(old) if old.is_dir() else old.unlink()
    shutil.copytree(eng, dst, dirs_exist_ok=True, ignore=SKIP)
    out = root / "WORK" / "outputs"
    s10 = here / "results" / "s10_outputs"
    s10.mkdir(parents=True, exist_ok=True)
    for f in (out / "s10").glob("*"):
        if f.is_file():
            shutil.copy2(f, s10 / f.name)
    fb, fbd = out / "freeze_bundle", here / "results" / "freeze_bundle"
    fbd.mkdir(parents=True, exist_ok=True)
    for f in fb.glob("*"):
        if f.name == "lexicon.json":
            (fbd / "lexicon.json.sha256").write_text(
                f"{sha(f)}  lexicon.json (embargoed; available to reviewers and on request under a data-use agreement)\n")
        elif f.is_file():
            shutil.copy2(f, fbd / f.name)
    if (out / "ledger_summary.json").exists():
        shutil.copy2(out / "ledger_summary.json", here / "results" / "ledger_summary.json")
    if (out / "data_provenance_log.csv").exists():
        shutil.copy2(out / "data_provenance_log.csv", here / "provenance" / "data_provenance_log.csv")
    for notes in (root / "run_notes", eng):
        for f in notes.glob("CHANGES*.md"):
            shutil.copy2(f, here / "docs" / "run_notes" / f.name)
    files = sorted(p for p in here.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256" and ".git" not in p.parts)
    (here / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(here).as_posix()}\n" for p in files))
    print(f"collected: frozen code, outputs and notes; {len(files)} files listed in MANIFEST.sha256")


if __name__ == "__main__":
    main()
