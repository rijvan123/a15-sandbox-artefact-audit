# A15: sandbox-artefact audit of deep-learning malware family attribution

Code, derived data and documentation for the article

> Beg, R. *Deep learning attribution of malware families from sandbox behaviour traces transfers poorly between installations of the same sandbox.* Submitted to *Scientific Reports* (2026).

The study audits whether deep-learning models that attribute malware to families from CAPEv2 sandbox traces transfer between two independently operated installations of the same sandbox (MELD-DS-448 and WinMET). Every trace token is classed as behaviour (B), footprint (F) or probing (P) by a frozen lexicon; hypotheses H1–H4 were pre-registered on the Open Science Framework (https://doi.org/10.17605/OSF.IO/K5EPU) before any test trace was scored; and every input file, model and result is sealed in a hash-chained ledger anchored with RFC 3161 time-stamp tokens.

**Main result.** Within each installation the Transformer reached 47.7 and 62.0 points of family-balanced macro-F1; on the other installation's traces of the same families it reached 23.3 and 24.1 points, and every learner lost 21–53 points. The measured results file is `results/results.json` (SHA-256 `b2e80138bcbc1217162c046110a9a62880ce7361aee7c5c3e6a2e0c0c3e7b692`, ledger entry 130,782).

## Contents

| Folder or file | What it holds |
|---|---|
| `s10_engine/` | The frozen acquisition, sealing, harness and evaluation code (package `a15`, `run_s9.py`, `run_s10.py`, tests) exactly as run, copied from the author's computer by `collect_release.py` and checked against `provenance/SHA256SUMS_s10_engine.txt` |
| `results/results.json`, `results/assembly_info.json` | Measured results of the evaluation and the assembly record of the test sets |
| `results/s10_engine_figures/` | Figures as written by the evaluation engine |
| `results/s9_outputs/` | Stage outputs of acquisition and sealing: ingest counts, field audit, hook set, harmonisation counts, overlap, dry run, alias review, ledger summary, run information |
| `figures_src/` | Code for every figure of the article: `s7_figures.py` (Figs 1–3), `figures_final.py` (Figs 6–8), `figures_extra.py` (Figs 4, 5 and 9–11), `figS1.py` (Supplementary Fig. S1) |
| `manuscript/` | Manuscript and Supplementary Information sources and the build script that assembles them from the results |
| `analysis/s8_checks.py` | Symbolic checks of the mathematical model |
| `tools/` | Ledger re-anchoring script and the amendment-3 checks |
| `provenance/` | Data Provenance Log and SHA-256 manifests of the frozen code and deliverables |
| `docs/` | Pre-registration text, deviation record D1 and the run notes of each stage |
| `logs/` | Logs of the acquisition and sealing stages, unit tests and the self-test |
| `verify_release.py` | Checks every digest in `MANIFEST.sha256`, the frozen code and the results file |

## Data

No dataset file is redistributed here; download them from their sources:

- **MELD-DS-448**: https://huggingface.co/datasets/MeldProject/MELD-DS-448 (commit e2dbf22; CC BY-NC-SA 4.0)
- **WinMET**: https://doi.org/10.5281/zenodo.12647555 (record 16414116; CC BY-NC 4.0)
- **EMBER2024** (family labels of shared SHA-256s only): Hugging Face `joyce8/EMBER2024` (revision 3d23efe; Apache-2.0)

Every downloaded file, with its source, version, licence, size and digest, is listed in `provenance/data_provenance_log.csv`. No malware binary is downloaded, unpacked or executed by any script in this repository.

**Embargoed material.** The token-level lexicon (`lexicon.json`, the frozen table of footprint and probing tokens) is not published, to avoid providing an evasion aid; `results/freeze_bundle/lexicon.json.sha256` records its digest. It is available to reviewers and, after publication, from the corresponding author for non-commercial research under a data-use agreement. The rules that build it are in `s10_engine/a15/lexicon.py`.

## Reproducing the study

Requirements: Windows Subsystem for Linux (Ubuntu 24.04) or Linux, a CPU (no graphics processor needed), about 60 GB of free disk space, and conda.

```bash
conda env create -f environment.yml && conda activate a15-s9      # or: pip install -r requirements.txt
cd s10_engine
python -m pytest -q tests                                          # unit tests of the harness and engine

# Acquisition, sealing and freeze (stage S9)
python run_s9.py --root WORK --tsa https://freetsa.org/tsr acquire
python run_s9.py --root WORK --tsa https://freetsa.org/tsr ingest --inst I1
python run_s9.py --root WORK --tsa https://freetsa.org/tsr ingest --inst I2
python run_s9.py --root WORK audit
python run_s9.py --root WORK harmonise
python run_s9.py --root WORK --tsa https://freetsa.org/tsr lexicon
python run_s9.py --root WORK --tsa https://freetsa.org/tsr overlap
python run_s9.py --root WORK --tsa https://freetsa.org/tsr dryrun
python run_s9.py --root WORK freeze-prep
python run_s9.py --root WORK --tsa https://freetsa.org/tsr freeze --confirm-reviewed

# Evaluation (stage S10; about 57 h of wall-clock time on 12 logical CPUs)
python run_s10.py --root WORK --tsa https://freetsa.org/tsr assemble
python run_s10.py --root WORK --tsa https://freetsa.org/tsr evaluate --workers 4 --threads 3
python run_s10.py --root WORK status --workers 4
```

The ledger requires a custodian key in `A15_CUSTODIAN_KEY` (see `docs/run_notes/README_S9_2026-11-02.md`). Stage details, amendments and the deviation record are in `docs/`.

**Figures and manuscript from the results**

```bash
python figures_src/figures_final.py results/results.json s10_figs          # Figs 6-8 in the article layout
cp results/results.json s10_figs/
python figures_src/figures_extra.py results/results.json results/assembly_info.json manuscript/figs
python figures_src/figS1.py results/assembly_info.json manuscript/figs/FigS1
cd manuscript && pip install -r requirements-build.txt
python build.py --out out --s10 ../s10_figs                                # Word and PDF files in manuscript/out
```

The manuscript build also needs pandoc and LibreOffice (`sudo apt install pandoc libreoffice-writer libreoffice-math`). It refuses any results file whose digest differs from the one recorded in the ledger.

## Verifying this release

```bash
python verify_release.py
```

checks every file against `MANIFEST.sha256`, the frozen code against `provenance/SHA256SUMS_s10_engine.txt`, and `results/results.json` against its ledger digest.

## Licences

- Code: MIT licence (`LICENSE`).
- Derived data in `results/`: CC BY-NC-SA 4.0 (`LICENSE-DATA.md`), as required by the licences of MELD-DS-448 and WinMET.

## Citation

See `CITATION.cff`. Please cite the article and the archived release (Zenodo DOI on the release page).

## Contact

Rijvan Beg, Department of Computer Science and Engineering, SRM University-AP, Amaravati 522240, India (rijvan.beg@gmail.com).
