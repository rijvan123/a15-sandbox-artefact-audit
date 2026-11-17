# s10_engine (frozen code)

This folder holds the code exactly as it ran for the study: the package `a15` (acquisition, CAPE parsing, ledger,
de-duplication, lexicon, masks, tokens, splits, models, probes, statistics, freeze and gates), `run_s9.py`,
`run_s10.py`, the evaluation runner of 7 November 2026, benchmarks and tests.

It is filled from the author's work folder by `collect_release.py`, which refuses to copy unless every file matches
its digest in `provenance/SHA256SUMS_s10_engine.txt`. Check it at any time with `python3 verify_release.py`.
