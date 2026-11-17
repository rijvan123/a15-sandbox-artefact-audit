# Results

- `results.json`: measured results of the evaluation (SHA-256 b2e80138bcbc1217162c046110a9a62880ce7361aee7c5c3e6a2e0c0c3e7b692, ledger entry 130,782).
- `assembly_info.json`: assembly record of the test sets (splits, near-duplicate search, kept traces per threshold).
- `s10_engine_figures/`: figures as written by the evaluation engine (the article uses the same data redrawn by `figures_src/figures_final.py`).
- `s9_outputs/`: outputs of the acquisition and sealing stage (counts, field audit, hook set, overlap, dry run, alias review, ledger summary at the end of S9, run information).
- `s10_outputs/`, `freeze_bundle/`, `ledger_summary.json`: added by `collect_release.py` (the token-level lexicon is embargoed; only its digest is included).

Licence: CC BY-NC-SA 4.0 (see `LICENSE-DATA.md`).
