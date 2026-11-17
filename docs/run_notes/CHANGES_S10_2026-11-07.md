

# S10 runner, 7 November 2026 (after the OSF registration, before any test trace was scored)

Context: the study was frozen on 5 November 2026 (freeze entry 130,774; anchor 130,775, freetsa.org) and registered
on OSF on 7 November 2026 (DOI 10.17605/OSF.IO/K5EPU). No frozen file changes. None of the 24 S9 code files changes
(SHA256SUMS_s10_engine.txt still verifies). The 2 November S10 engine (from `s10_engine.zip`) was adapted to the
frozen plan and to the analysis machine; its earlier files are kept in `run_notes/s10_engine_2nov_as_extracted/`.

## Changed or new files
- `run_s10.py`: evaluate is restartable and can use parallel workers; reads frozen inputs from the freeze bundle and
  checks their digests; writes the OSF DOI, plan, code digests, software versions and the assembly digest into the
  `test_score` ledger entry; refuses a restart with another plan or assembly; `assemble` command (test sets and
  cells only, with an `s10_assembly` ledger entry, so the kept test traces are known before any training);
  `status` command; label "MEASURED
  (authors' PC ...)" replaces "MEASURED on Cowork"; the superseded cpu-reduced plan of 2 November is removed;
  `fixture-test` plan for code tests on the S9 fixtures only (labelled in every output).
- `a15/audit.py`: BiLSTM triage (S10-13); shared length probe (S10-14); `summarise()` shared by the in-process and
  job paths; H1 "not tested" when a learner has no artefact-only probe; P9 from stored predictions with an exact
  Kendall test that also handles tied months; `LGBM_THREADS` for parallel workers.
- `a15/s10data.py` (new): streamed assembly (S10-11, S10-12), frozen-input check, `nn_similarities_stream` (equal to
  `dedup.nn_similarities`), content digest of the assembly.
- `a15/s10jobs.py` (new): jobs, dependencies, checkpoints, worker pool, time estimate, collection of results.
- `a15/figures.py`: margin lines at DELTA (3 points) instead of 2.
- `bench_parallel.py` (new): throughput of workers x threads on the analysis machine (no data).
- `tests/test_s10_jobs.py` (new): the job engine (inline, interrupted and restarted, and two spawned workers) gives
  exactly the in-process results; job counts per cell equal the K6 plan (39; BiLSTM 10); `nn_similarities_stream`
  equals the reference; same-binary clusters; compact dtypes leave training unchanged; exact Kendall test.
- `README_S10.md`: rewritten for the evaluation on the authors' PC.

## Analysis decisions added before scoring (recorded in the ledger with the code digests)
- S10-11: every retained CAPE report is a trace (as counted at S9); a binary's reports share its split part (S9 rule:
  test, then validation, then training); reports of one binary in a test set form one bootstrap cluster; a test
  report is dropped when its binary has a report in that origin's training union (both installations).
- S10-12: P9 series from MELD-DS-448 traces of the origin-1 families first seen March-August 2025, de-duplicated
  against the origin-1 training union, scored by the five origin-1 base models.
- S10-13: BiLSTM unmasked and F-masked (p = 1) only, configuration = first draw of the registered search for the
  origin, no artefact-only probe; H1 not tested for the BiLSTM (frozen compute plan of amendments 3 and 5).
- S10-14: one length-only probe per cell and seed, shared by all learners (identical values).
- Training length: at most 20 epochs with early stopping after 3 epochs without improvement in validation macro-F1
  (the S7 protocol and the default of `train_deep`); the K6 projection had assumed 10 epochs per run.

## Verification (cloud workspace, 7 Nov 2026)
- 50 of 50 tests pass (44 earlier, 6 new).
- S9 fixture G5 (two installations, frozen with the local development time-stamp authority): the guard refused a
  tampered frozen file, an untrusted anchor without the test flag and a restart with another plan; the evaluation
  with the fixture-test plan was stopped after 20 jobs, restarted with two workers, and completed with results,
  figures, a `results` ledger entry and an anchor; the ledger verifies. Second run: `assemble` wrote the
  `s10_assembly` entry, `evaluate` reused that assembly (same digest in `test_score`), `assemble` was refused once
  scoring had begun, and `status` reported the jobs. The two WinMET cells with identical family sets shared one fit.
  On this fixture the antimode rule gave tau_J 0.555-0.558 and the near-duplicate rule kept 17-26 test traces of
  84-120 per set: the fixture repeats a few hundred reports, so it says nothing about the real data, but it shows why
  `assemble` reports the kept counts before any training.
- Synthetic rehearsals with the job engine (2 workers; n1 100, n2 80, 2 configurations, 2 seeds, 5 epochs, 999
  replicates). Planted: transfer gaps psi3 of 9.6-24.7 points across learners; the Transformer's H1 and H2 supported,
  H3 inconclusive (I2 lower bound 2.0), H4 not tested. Null: Transformer gaps -5.3 and 0.4 points, H3 inconclusive,
  H4 not tested; the 1D-CNN's H3 rejected. The planted run was interrupted at 146 of 148 jobs and finished after a
  restart.


# Deviation D1, 8 November 2026 (after the first S10 assembly, before any model was trained or scored)

Context: the first `assemble` on the authors' PC (7 Nov 2026; ledger action `s10_assembly`; 4,599 s, 1.19 GB peak)
showed that the registered antimode rule gave tau_J = 0.535, 0.532 and 0.528. At these thresholds it kept 167 of
1,001, 127 of 848 and 157 of 649 shared-family MELD-DS-448 test traces, and 32 of 1,001, 34 of 1,018 and 31 of 1,007
WinMET test traces. The authors chose tau_J = 0.8 as the primary threshold and reported the rest as sensitivity
analyses (`deliverables/DEVIATION_D1_2026-11-08.md`). No frozen file changes. The 24 S9 code files do not change.
The previous versions of the changed files are in `run_notes/s10_engine_before_D1/`.

## Changed files
- `a15/audit.py`:
  - `restrict_cell()` and `restrict()` select an analysis's kept test traces (equation 9 at its tau_J, plus the
    same-binary rule), the matching prediction columns and the monthly P9 traces;
  - `analyse_all()` runs the primary analysis (Plan.primary = "0.8") in full and the other entries of Plan.analyses
    ("registered", "0.7", "0.9") as sensitivity analyses of the same predictions;
  - IG, referral and the environment probe run on the primary analysis's traces;
  - P5, P8 and P9 are recomputed for every analysis; per-trace matched-mask counts are kept for P8 and are not
    written to results.json.
- `a15/s10data.py`:
  - cells hold every shared-family test trace with its largest Jaccard index to the origin's training union and a
    same-binary flag;
  - test sets carry clusters at each analysis's tau_J (WinMET at 0.8 throughout);
  - `nn_profile` (histogram and kernel density with peaks and minima) for the Supplementary Information;
  - kept counts at every tau_J;
  - `ASSEMBLY_FORMAT`, so an assembly of an earlier engine is rebuilt, not reused.
- `a15/s10jobs.py`: the explain job uses the primary analysis's test traces.
- `a15/figures.py`: `TableS_tau_sensitivity.md/.csv`, which gives H1-H4 at every tau_J.
- `run_s10.py`:
  - `DEVIATIONS` record;
  - `record_deviations()` writes a `deviation` ledger entry with evidence ID D1 and anchors it with the time-stamp
    authority. It is refused once scoring has begun;
  - `assemble` and `evaluate` call it;
  - `test_score` lists the deviation entries;
  - `assembly_summary` prints the kept counts at every tau_J;
  - the Results draft states the primary threshold and has a sensitivity section.
- `tests/test_s10_jobs.py`:
  - the job-engine equivalence test now runs with near-duplicate data (four analyses);
  - new tests: `restrict` keeps traces and predictions aligned (and is idempotent); the deviation is recorded once and
    refused after scoring.
- `README_S10.md`: D1, commands and outputs.

## Verification (cloud workspace, 8 Nov 2026)
- 52 of 52 tests pass.
- S9 fixture G5, from its frozen state:
  - `assemble` wrote `s10_assembly`, `deviation` D1 and an anchor;
  - the registered analysis kept exactly the traces of the 7 November engine (20/84, 23/90, 17/102, 26/120, 19/108,
    25/120);
  - `evaluate` (fixture-test plan, 2 workers, 130 jobs) wrote `test_score` with D1 (entry and digest), results with
    four analyses, Table 2, `TableS_tau_sensitivity`, Figs. 4-6, a `results` entry and an anchor; the ledger
    verifies;
  - P9 and P8 differed by analysis as expected;
  - `assemble` was refused after scoring;
  - a copy holding a 7 November assembly was detected and assembled again before D1 was recorded.
