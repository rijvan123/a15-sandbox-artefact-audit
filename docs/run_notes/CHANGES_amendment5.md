

# Amendment 5 before the freeze, 5 November 2026: margin 3 points (K5 action) and the registered seeds and search

## Corrected dry run (amendment 4 code; authors' PC, 5 Nov 2026, 18:29-18:59 UTC, A15_THREADS 8 of 12 logical CPUs)
- Thread check (`bench_threads.py`): CPU plan of amendment 3 at 1, 2, 4, 6, 8 and 12 threads: 67, 41, 48, 40, 37
  and 41 h; fastest 8 threads.
- Dry run: 10 of 10 epochs; mean within-installation recall 0.459 (0.351 on 4 Nov); 8 of 9 families predicted within
  and 7 across; dry-run gap 17.2 points (95% CI 10.2-24.5) on the pseudo-origin (planning number, not a result).
- P1 holds (IG exactly 0 at masked positions). P4 holds: measured half-width 7.12 points, formula 6.04, ratio 1.18.
- K5 triggered: projected half-width 5.80 points at the smallest origin (n_eff 789); N_req 6,642 traces per test set
  at 2 points. Averaged over the three MELD-DS-448 origins (S8: averaging divides the within-installation variance by
  O; WinMET's single test set does not) the projection is about 4.3 points (estimated from the same inputs).
- K6: per trace at L 256, mean over 8 configurations: Transformer 21.0 ms, BiLSTM 38.5 ms, 1D-CNN 2.2 ms. CPU plan of
  amendment 3: 32.8 h (Transformer 20.1, BiLSTM 10.5, 1D-CNN 2.1); full factorial for all three: 59.1 h.

## Authors' decisions (5 Nov 2026)
1. K5: apply the action registered at S6. The test windows cannot widen (the three two-month windows cover March to
   August 2025 and MELD-DS-448 ends in August 2025); every family is kept; the margin Delta becomes 3 points before
   scoring. The projected half-width still exceeds 3 points, so equivalence and non-inferiority verdicts may be
   inconclusive; the registration says so. Delta is the smallest effect of interest in all of H1-H4, so the support
   thresholds also become 3 points.
2. Compute: restore the five registered seeds (11, 23, 37, 41, 53) and 20 search configurations for the Transformer
   (confirmatory) and the 1D-CNN; BiLSTM unmasked and F-masked only (5 seeds); n-gram LightGBM full factorial;
   sequence cap 256. Projected from the corrected dry run: Transformer 37.4 h, BiLSTM 17.6 h, 1D-CNN 3.9 h, about
   59 h in total (one run at a time, 8 threads, 10 epochs per run).

## Change
- `a15/config.py`: DELTA 3.0 (was 2.0); SEEDS (11, 23, 37, 41, 53) (amendment 3: first three); N_SEARCH 20
  (amendment 3: 8). K5_MAX_HALF_WIDTH stays 2.0, the registered K5 limit.
- `a15/freeze.py`: H1-H4 texts take the margin from DELTA (they said 2 points); SETTINGS records `k5_applied` and the
  compute plan of amendments 3 and 5.
- `a15/gates.py`: the K5 record adds `margin_in_force` and `projected_exceeds_margin`.
- `run_s9.py`, `bench_threads.py`: runs per cell from the configuration (39; BiLSTM 10); plan text.
- `tests/test_harness.py`: 40 tests (1 new: margin, seeds, search and records).

## Label-source check, names (diagnostic, no rule changed)
`K4_I1_meld_in_cape` showed 0% for loki (46 traces) and snakelogger (225) because CAPE names them "lokibot" and
"snake". Read from the sealed MELD-DS-448 records (5 Nov 2026; all records, the 22 curated out included): loki 46 of 46 "lokibot"; snakelogger 222 of 225
"snake" (1 redline, 1 redline and snake, 1 guloader); amadey 21 of 31 "amadey" (8 lumma). CAPE names a family in
6,125 of 26,144 MELD-DS-448 traces, so most MELD labels are not CAPE output; the limitation is the 54.5% ClarAVy
agreement (K4). The ClarAVy rule outcome does not change (loki had no confident ClarAVy label).

## Verification (cloud workspace, 5 Nov 2026)
40 of 40 tests pass; fixture dry run and freeze-prep complete with 20 configurations timed, runs per cell 39 and 10,
K5 margin 3, and hypothesis texts at 3 points.

## Stages to re-run
`dryrun` (K6 projection for the new plan; about 32 min) and `freeze-prep`. The hook set, lexicon, P3, splits, family
sets and alias table do not change.

## New SHA-256 (all 24 digests in ../SHA256SUMS_s10_engine.txt)
- a15/config.py 1db1354a0dc33249e09d6e63f53b45631f456b45432fb74ecfc7d0f5f28b8d62
- a15/freeze.py 7b66c44718b9c76cd4774322bb48e8b1c3da9fb2bd88327e66e45b72e3de736d
- a15/gates.py e798f87ade42eb3db6197962a13088b280a8a74d4af00c3ddf38b88c14aa8d71
- run_s9.py e03c069e97391fa0cc3a0c71f26dcce6ff655b45abfa0a5ef84683fda9bc7a52
- bench_threads.py d17102d94e6093497c2860185f1c0bc4a6962c137215038d6d249b9c2c601dba
- tests/test_harness.py ed61afaa27ac2b6bdf8ee876f6e961943c0402d73f71f013ab554404d67180ae
- README_S9.md 15245276999dba164a058810e11ffb9284729fa6610151057ed648ebfb29ba71
