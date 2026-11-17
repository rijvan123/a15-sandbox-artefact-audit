# Deviation D1 from the registered protocol: near-duplicate threshold

A15 sandbox-artefact audit · OSF registration DOI 10.17605/OSF.IO/K5EPU (registered 7 November 2026) · decided by the
authors on 8 November 2026 (India Standard Time), before any model was trained or any test trace scored.

## Registered rule

S8 equation (9) removes a test trace when its largest Jaccard index (API 3-gram sets) to any training or validation
trace of its origin's training union is at least tau_J. tau_J is the antimode of the kernel density of
nearest-neighbour similarities on that union, searched on [0.5, 0.95], with 0.8 as the fallback. The registration
uses 0.8 for WinMET's grouped split (S9) and for clustering WinMET test traces (S10-3). The S8 plan named 0.7 and 0.9
as sensitivity values.

## What the authors saw before deciding

They saw only the counts printed by the first S10 assembly (7 November 2026, ledger action `s10_assembly`). No model
had been trained and no test trace had been scored.

| Origin | Registered tau_J | MELD-DS-448 test traces kept | WinMET test traces kept |
| --- | --- | --- | --- |
| 1 | 0.535 (antimode) | 167 of 1,001 (83% removed) | 32 of 1,001 (97% removed) |
| 2 | 0.532 (antimode) | 127 of 848 (85% removed) | 34 of 1,018 (97% removed) |
| 3 | 0.528 (antimode) | 157 of 649 (76% removed) | 31 of 1,007 (97% removed) |

No test trace was removed by the same-binary rule (S10-11).

## Counts at every threshold (second assembly, 8 November 2026, after the decision and before any scoring)

| Origin | Kept MELD-DS-448: antimode / 0.7 / **0.8** / 0.9 | Kept WinMET: antimode / 0.7 / **0.8** / 0.9 |
| --- | --- | --- |
| 1 | 167 / 418 / **705** / 955 of 1,001 | 32 / 204 / **322** / 838 of 1,001 |
| 2 | 127 / 343 / **589** / 816 of 848 | 34 / 208 / **329** / 855 of 1,018 |
| 3 | 157 / 313 / **508** / 633 of 649 | 31 / 205 / **324** / 845 of 1,007 |

The antimode counts repeat those of the first assembly exactly. On each origin's training union, the
nearest-neighbour similarity density has peaks near 0.47 and 0.89 and its minimum at 0.53 (Supplementary
Information, from `assembly_info.json`).

## Change

1. Every model scores every shared-family test trace. The near-duplicate rule is applied when the results are
   analysed, not when the test sets are built.
2. The primary analysis applies equation (9) with tau_J = 0.8, the registered fallback.
3. Three sensitivity analyses use the same models and predictions: the registered antimode thresholds (0.535, 0.532
   and 0.528), tau_J = 0.7 and tau_J = 0.9. Each has its own decisions for H1–H4.
4. P1, P2, P6, P7 and the environment-identity probe are computed on the primary analysis's test traces. P5 (design
   effect), P8 (matched-mask borrowing) and P9 (monthly decline) are computed for every analysis.

## Unchanged

The following are unchanged:

- training and validation sets, and family sets;
- models, random searches, seeds and masks;
- estimands and the cluster bootstrap;
- decision rules, Holm adjustment and the H4 gate;
- the same-binary rule (S10-11).

## Reason

At about 0.53 the rule removed most of each test set. That left 31–34 WinMET and 127–167 MELD-DS-448 test traces per
origin, too few to estimate family-balanced macro-F1 over the shared families with useful precision. The rule exists to
remove near-copies of training traces. The fallback 0.8 is the registration's own threshold for that purpose
elsewhere (WinMET split and clusters).

## Record

- **Ledger:** the `run_s10.py assemble` command writes a `deviation` entry with evidence ID D1. Its `d_f` is the
  SHA-256 of the deviation record, and its payload is the record itself. The entry is anchored at once with a trusted
  time-stamp (freetsa.org). The first `test_score` entry comes later in the chain and lists D1 with its entry number
  and digest. On the authors' PC, D1 is ledger entry 130,778 (digest 063e8eab...). It was anchored in entry 130,779
  by freetsa.org at 8 November 2026 01:39:20 UTC, in batch 130,776-130,778, which also holds both `s10_assembly`
  entries.
- **Outputs:**
  - `results.json`: `primary_analysis`, and `sensitivity` with the three other analyses;
  - `TableS_tau_sensitivity.md/.csv`: every estimate, interval and verdict at each threshold;
  - Results draft: a sensitivity section;
  - `assembly_info.json`: the nearest-neighbour similarity density (`nn_profile`) for a Supplementary figure at S12.

## Text for the OSF registration update

> Deviation D1 (decided 8 November 2026, before any model was trained or scored). The registered rule set the
> near-duplicate threshold tau_J to the antimode of the nearest-neighbour Jaccard similarity density. It gave tau_J =
> 0.535, 0.532 and 0.528 at the three origins and classed 76–85% of the MELD-DS-448 and 97% of the WinMET
> shared-family test traces as near-duplicates. That left 31–34 WinMET test traces per origin. The primary analysis
> therefore uses the registered fallback tau_J = 0.8. The registered thresholds, 0.7 and 0.9 are reported as
> sensitivity analyses of the same models, each with its own H1–H4 decisions. Nothing else changes. The decision is
> recorded in the evidence ledger (entry 130,778), with a trusted time-stamp (freetsa.org, 8 November 2026 01:39:20
> UTC), before the first test score.

## Text for the manuscript (Methods, "Deviations from the registration")

> The registered rule set the near-duplicate threshold to the antimode of the nearest-neighbour Jaccard similarity
> density on each origin's training union. Before any model was trained or scored, it returned tau_J = 0.535, 0.532
> and 0.528. At these thresholds it classed 76–85% of the MELD-DS-448 and 97% of the WinMET shared-family test traces
> as near-duplicates, leaving 31–34 WinMET test traces per origin. We therefore used the registered fallback
> tau_J = 0.8 as the primary threshold. We report the registered thresholds, 0.7 and 0.9 as sensitivity analyses of the
> same models (Supplementary Table S_tau). The decision was recorded, with a trusted time-stamp, in the evidence ledger
> before the first test trace was scored.
