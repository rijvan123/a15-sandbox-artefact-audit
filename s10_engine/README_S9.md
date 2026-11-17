# S9 harness: A15 sandbox-artefact audit (data acquisition, sealing, gates, freeze)

Code for stage S9 of the Scientific Reports manuscript on behavioural malware evidence. It acquires and seals the
two CAPEv2 installations (MELD-DS-448 = I1, WinMET = I2), audits their fields, harmonises labels and hook sets,
builds and seals the B/F/P lexicon, tests prediction P3 (installation overlap by token class), counts shared-family
traces (K3, N_req), checks labels (K4), runs the dry run (K5, K6, P1, P4) and prepares the freeze and the OSF
registration. No test window is opened before `freeze`.

## Layout
- `a15/config.py` every constant fixed at S6-S8 and the dataset specifications (pinned revisions, skip patterns)
- `a15/ledger.py` hash chain, RFC 6962 Merkle roots, RFC 3161 tokens through OpenSSL, verification, V_frz
- `a15/acquire.py` pinned downloads with published-digest checks, Data Provenance Log, substitution entries
- `a15/cape.py` streaming of split and password-protected 7-Zip archives (one report in memory at a time),
  report parsing into trace, environment and label records, MALVADA's curation criteria
- `a15/tokens.py` class tokeniser (closed enumerations only), run-length collapse, head-plus-tail cut, L rule, vocabulary
- `a15/lexicon.py` documented B/F/P rules (ATT&CK T1497; CAPE behaviour), boundary rules with an alternative class
- `a15/dedup.py` API 3-gram MinHash and LSH, exact-Jaccard decisions, antimode rule for tau_J, test clusters
- `a15/masks.py`, `a15/overlap.py`, `a15/splits.py`, `a15/stats.py`, `a15/models.py`, `a15/probes.py`,
  `a15/gates.py`, `a15/freeze.py` equations 3-28 and S1-S9 of S8
- `a15/labels.py` K4 label checks: ClarAVy agreement on shared SHA-256s (EMBER2024 archives read without unpacking),
  CAPE-label comparisons and the S6 rule that keeps only agreeing families (amendment 3)
- `run_s9.py` driver; `plots_s9.py` diagnostics figure; `bench_threads.py` thread-count check for K6 (amendment 4);
  `tests/test_harness.py` 40 tests

## Install
`pip install -r requirements.txt` (Python 3.12; CPU). OpenSSL must be on the path. Tested versions are pinned.

## Run in Cowork (one stage per task, in this order)
```
export A15_CUSTODIAN_KEY=<secret held by the custodian>      # HMAC key for actor pseudonyms; never logged
python run_s9.py --root WORK --tsa https://freetsa.org/tsr acquire       # I1 at commit e2dbf22, latest WinMET record
python run_s9.py --root WORK --tsa https://freetsa.org/tsr ingest --inst I1
python run_s9.py --root WORK --tsa https://freetsa.org/tsr ingest --inst I2
python run_s9.py --root WORK audit
python run_s9.py --root WORK --tsa https://freetsa.org/tsr ember-labels   # K4: EMBER2024 ClarAVy labels (about 23 GB)
python run_s9.py --root WORK harmonise      # writes alias candidates; authors review WORK/outputs/alias_table.json
python run_s9.py --root WORK --tsa https://freetsa.org/tsr lexicon   # sealed before any overlap statistic
python run_s9.py --root WORK --tsa https://freetsa.org/tsr overlap   # prediction P3
python run_s9.py --root WORK --tsa https://freetsa.org/tsr dryrun    # K5, K6, P1, P4 on the pseudo-origin (10 epochs)
python run_s9.py --root WORK freeze-prep    # freeze bundle and OSF_registration.md for the authors
python run_s9.py --root WORK --tsa https://freetsa.org/tsr freeze --confirm-reviewed
```
The stages after `ingest` read the sealed records one at a time (amendment of 3 November 2026), print a progress
line with memory every 2,000 records and write `outputs/runinfo_<stage>.json` (time, peak memory and, for
`harmonise`, the near-duplicate clustering counts). On a 2,500-record test fixture each of `audit` to `overlap`
peaked below 0.2 GB and the dry run at 1.8 GB (mostly PyTorch), against 2.3-4.1 GB for the earlier all-in-memory
code, whose dry run was stopped at 8 GB.
Amendment 3 (4 November 2026, before the freeze): `ember-labels` adds the pre-registered ClarAVy check for K4 and
`harmonise` drops families that fail it (S6 rule) and applies the authors' alias table to every K4 comparison; the dry
run uses the shared families with all their pseudo-origin traces and WinMET traces from the training and validation
parts only; and the CPU plan replaces the GPU-free full design (sequence cap 256, 8 search configurations, 3 seeds,
BiLSTM unmasked and F-masked only). After amendment 3 the stages to re-run are `ember-labels`, `harmonise`, `dryrun`
and `freeze-prep`: the hook set, the lexicon and P3 do not change.
Amendment 4 (5 November 2026, before the freeze): the dry-run model trains for all requested epochs (10 by
default, the epochs the K6 projection assumes; on 4 November it stopped after 3 of 5 because early stopping fired
without a validation set), and the K6 input is the mean cost over the 8 search configurations, recorded with the
thread count and load average. `python bench_threads.py` times the learners at several thread counts (no data
read, nothing written); set `A15_THREADS` to the fastest, since `a15/models.py` otherwise uses every logical CPU.
Then re-run `dryrun` and `freeze-prep`.
Amendment 5 (5 November 2026, before the freeze; the authors' decisions after the corrected dry run): the margin
Delta is 3 points, the K5 action registered at S6 (the projected half-width was 5.8 points and the test windows
cannot widen), and the five registered seeds and 20 search configurations are restored for the Transformer and
1D-CNN (BiLSTM unmasked and F-masked only); the corrected dry run projected about 59 h for this plan. Re-run
`dryrun` and `freeze-prep`.
Set `A15_TSA_CA` and `A15_TSA_CERT` to the TSA's CA and signing certificates so tokens verify. Disk: about
17.4 GB for the I1 report and sequence archives plus about 13 GB for WinMET; nothing is unpacked to disk.
The ASM disassembly, Markdown and split-sequence archives of MELD-DS-448 are listed in the provenance log and never
downloaded. No binary is downloaded, unpacked or executed at any stage.

## Sandbox self-test (claude.ai, 1 CPU, 3 GB RAM; measured on sandbox)
`python run_s9.py --root RUN selftest --reports-archive malvada/test_reports/test_reports.7z --malvada-dir malvada`
uses the 200 raw CAPE reports shipped with MALVADA (commit 2497402; published MD5 matched). Results are in
`s9_selftest_report.json`; they test the code on real reports and are not study results.

## Known limits
The Cowork stages reuse the unit-tested functions but were not executed end to end, because the sandbox cannot reach
Hugging Face, Zenodo, Google Drive or a public time-stamping authority. The WinMET label-file layout and the
EMBER2024 metadata route for K4 are confirmed at the `audit` and `harmonise` stages. 7-Zip streaming is pinned to
py7zr 0.22.0. Anchors made without an external TSA use a local development authority and are marked untrusted.
