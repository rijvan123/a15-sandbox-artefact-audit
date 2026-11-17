# S10 evaluation engine (A15 audit), version of 8 November 2026 (deviation D1)

S10 scores the frozen study: the models of Algorithm 4 for every learner, installation and origin; the estimands of
H1-H4 with the cluster bootstrap; the decision rules; the checks P1, P2 and P5-P9; Table 2, Figs. 4-6 and a Results
draft whose numbers are inserted by code. It runs on the authors' PC (WSL Ubuntu, CPU only).

**Deviation D1 (8 Nov 2026, before any scoring; `deliverables/DEVIATION_D1_2026-11-08.md`).** Every model scores
every shared-family test trace. The near-duplicate rule (equation 9) is applied at analysis time: the primary analysis
uses tau_J = 0.8 (the registered fallback); the registered antimode thresholds, 0.7 and 0.9 are sensitivity analyses
of the same models. `assemble` writes the deviation to the ledger and time-stamps it before the first test score.

It refuses to score unless the ledger verifies, the freeze entry sits in a batch anchored by a trusted time-stamp
(V_frz = 1) and every file in `outputs/freeze_bundle/` matches its digest in the freeze entry. The OSF registration
(DOI 10.17605/OSF.IO/K5EPU) is written into the first S10 ledger entry with the plan, the code digests and the digest
of the assembled test sets.

## What changed since the 2 November engine

| ID | Change | Why |
| --- | --- | --- |
| S10-10 | Training runs as jobs (one model each) that write their result to disk; the run restarts where it stopped; jobs can run in parallel worker processes | A run of several days must survive sleep, restarts and closed windows; several processes use 12 CPUs better than one |
| S10-11 | Every retained CAPE report is a trace (as in the S9 counts); a binary's reports share its split part; reports of one binary form one bootstrap cluster; a test report is dropped if its binary has a report in that origin's training union | The 2 November assembly kept one report per SHA-256 and could repeat a test trace; S9 counted reports |
| S10-12 | P9 series: MELD-DS-448 traces of the origin-1 families first seen March-August 2025, de-duplicated against the origin-1 training union | The 2 November engine built the series only for synthetic data |
| S10-13 | BiLSTM: unmasked and F-masked (p = 1) models only, configuration = first draw of the registered search, no artefact-only probe, so H1 is "not tested" for the BiLSTM | Compute plan of amendments 3 and 5, frozen in settings.json |
| S10-14 | The length-only probe is fitted once per cell and seed and shared by all learners | It does not depend on the learner; values are unchanged |
| — | Records are streamed (7 GB machine); frozen inputs are read from the freeze bundle and checked; token classes come from the frozen lexicon table; figures draw the 3-point margin; P9's exact Kendall test also handles tied months | Amendment 2 pattern; integrity; amendment 5 |
| D1 | All shared-family test traces are scored; analyses at tau_J = 0.8 (primary), the registered antimode, 0.7 and 0.9; IG, referral and the environment probe on the primary analysis's traces; P5, P8 and P9 per analysis; `TableS_tau_sensitivity`; an assembly of an earlier engine is rebuilt | The registered antimode (0.528-0.535) removed 76-85% of MELD-DS-448 and 97% of WinMET test traces (first assembly, no model trained) |

The rules S10-1 to S10-9 of 2 November stand (de-duplication against the union of both installations' training
traces, one bootstrap over all test sets, WinMET clustered at tau_J = 0.8, Holm then the H4 gate, shared WinMET fits
for identical family sets, LightGBM dropout as one draw per trace, referral at the 95th percentile, untrusted anchors
refused, Results draft from results.json).

## Commands (WSL Ubuntu terminal)

```bash
source ~/a15/a15env.sh     # sets $PY (the study's Python) and $W (the WORK folder), A15_TSA_CA/CERT, A15_THREADS,
                           # and moves into s10_engine/

# 1. Tests (about 5-8 minutes): 52 tests, including the job engine against the in-process engine
$PY -m pytest -q tests

# 2. Speed check (about 6 minutes; reads no data): which workers x threads trains fastest on this PC
$PY bench_parallel.py 2>&1 | tee $W/s10_bench_wsl.log

# 3. Assemble the test sets and cells (about 80 minutes; no model is trained or scored). Prints the registered tau_J
#    and how many test traces each tau_J keeps per test set and origin; writes an `s10_assembly` ledger entry and the
#    `deviation` entry D1, anchored with a trusted time-stamp. An assembly made by an earlier engine is rebuilt.
$PY run_s10.py --root $W --tsa https://freetsa.org/tsr assemble 2>&1 | tee $W/s10_assemble_d1_wsl.log

# 4. The evaluation (days; restartable). Use the workers and threads that step 2 printed.
$PY run_s10.py --root $W --tsa https://freetsa.org/tsr evaluate --workers 4 --threads 3 \
    2>&1 | tee -a $W/s10_evaluate_wsl.log

# Progress and time left, from a second terminal at any time (after `source ~/a15/a15env.sh`)
$PY run_s10.py --root $W status --workers 4
```

After any interruption (sleep, restart, closed window, Ctrl-C), run the step 4 command again: finished jobs are kept,
the ledger gets an `s10_resume` entry, and the plan and assembled test sets must match those recorded at the start.
Keep Windows from sleeping while it runs (Settings > System > Power > Sleep: Never, while plugged in).

`evaluate` uses the assembly of step 3 (checked against its ledger entry), writes the `test_score` ledger entry (OSF
DOI, plan, code digests, assembly digest) and starts the jobs. Without step 3 it assembles first.

## Outputs

- `WORK/outputs/s10/`: `results.json` (primary analysis, tau_J = 0.8, plus `sensitivity` for the other three),
  `Table2_H1_H4.md` and `.csv` (primary), `TableS_tau_sensitivity.md` and `.csv` (all four analyses), `Fig4-6` (PDF
  and 300 dpi PNG; primary), `results_draft.md`. The ledger gets a `results` entry with their SHA-256 and an anchor
  (trusted time-stamp).
- `WORK/work/s10/`: `assembly.pkl` (test sets and cells), `assembly_info.json` (registered tau_J, nearest-neighbour
  similarity profile, kept test traces at every tau_J, L and vocabulary per cell), `jobs/` (one file per model; seed-1
  deep models as `.pt`), `progress.jsonl` (time, epochs and threads of every job).

Do not edit or delete anything in `WORK/work/s10/` or `WORK/outputs/freeze_bundle/` while S10 runs or before the
results are archived.

## Rehearsal on synthetic data (any machine; never results)

```bash
$PY run_s10.py --root ~/a15_rehearsal rehearsal --scenario planted --engine jobs --workers 2
$PY run_s10.py --root ~/a15_rehearsal rehearsal --scenario null --engine jobs --workers 2
```
