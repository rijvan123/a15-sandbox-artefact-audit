# Results draft (numbers inserted by run_s10.py; wording to be finalised at S12)

Confirmatory learner: Transformer. Estimates are family-balanced macro-F1 points averaged over origins and 5 seeds; intervals are 95% cluster-bootstrap percentile intervals (9,999 replicates); P values are one-sided bootstrap P values, combined by the intersection-union rule and adjusted by Holm's method over H1-H4.

Test traces were kept when their largest Jaccard index to the training union of their origin was below tau_J = 0.8 and their binary had no report in that union (deviation D1 from the registered antimode rule, recorded before any test trace was scored; Methods): MELD-DS-448 705 of 1,001, 589 of 848, 508 of 649; WinMET 322 of 1,001, 329 of 1,018, 324 of 1,007 shared-family test traces at origins 1, 2, 3.

## Artefact tokens alone identify families

In MELD-DS-448, the artefact-only probe reached 40.2 points with behaviour tokens masked, against 32.0 for the length-only probe: a difference of 8.2 (95% CI 4.4 to 11.5) points.
In WinMET, the artefact-only probe reached 50.7 points with behaviour tokens masked, against 48.4 for the length-only probe: a difference of 2.4 (95% CI -6.9 to 9.8) points.
H1 was inconclusive (P = 0.6091, Holm-adjusted P = 1.0000).

## Masking footprint and probing tokens

In MELD-DS-448, masking F and P tokens at test lowered macro-F1 from 47.7 to 43.9 points, whereas a matched random mask of the same size left 43.9; the excess drop was -0.0 (95% CI -1.0 to 0.9) points.
In WinMET, masking F and P tokens at test lowered macro-F1 from 62.0 to 57.3 points, whereas a matched random mask of the same size left 56.8; the excess drop was -0.5 (95% CI -2.2 to 1.3) points.
H2 was rejected (P = 1.0000, Holm-adjusted P = 1.0000).

## Transfer between installations

Models trained on MELD-DS-448 scored 47.7 points within and 23.3 on WinMET, a gap of 24.4 (95% CI 20.5 to 27.6) points.
Models trained on WinMET scored 62.0 points within and 24.1 on MELD-DS-448, a gap of 38.0 (95% CI 22.6 to 42.2) points.
H3 was supported (P = 0.0001 (floor), Holm-adjusted P = 0.0004).

## Training without footprint tokens

F-masked training recovered a share rho = 0.046 (95% CI 0.014 to 0.084) of the pooled gap and changed within-installation macro-F1 by -0.2 (95% CI -1.3 to 0.7) points in MELD-DS-448 and -0.9 (95% CI -2.9 to 1.4) in WinMET. H4 was rejected (P = 0.9958, Holm-adjusted P = 1.0000).

## Explanations, referral and environment identity

Integrated Gradients gave exactly zero attribution at masked positions in every cell (P1 held); the completeness criterion was met with m = 50 nodes. The base model's mean footprint share of attribution was 3.1%. The referral rate on within-installation test traces ranged from 3.5% to 8.0% against the 5% budget.
In I1_o1, a probe on frozen embeddings identified the analysis machine with a balanced accuracy of 11.8% against 10% chance (permutation P = 0.089).
In I1_o2, a probe on frozen embeddings identified the analysis machine with a balanced accuracy of 10.4% against 10% chance (permutation P = 0.404).
In I1_o3, a probe on frozen embeddings identified the analysis machine with a balanced accuracy of 14.5% against 10% chance (permutation P = 0.003).

## Secondary learners and checks

BiLSTM (secondary, unadjusted): psi2 = 2.0 and 1.0 points; transfer gaps 21.4 and 41.7; rho = -0.020.
1D-CNN (secondary, unadjusted): psi2 = -0.0 and 1.7 points; transfer gaps 27.3 and 47.4; rho = 0.004.
n-gram LightGBM (secondary, unadjusted): psi2 = 2.9 and 18.4 points; transfer gaps 24.7 and 52.8; rho = 0.045.
Design effects of the test sets: I2_g 229.59, I1_o1 4.56, I1_o2 4.63, I1_o3 2.56 (P5 bound 1.2). Matched random masks borrowed at most 14.3% of positions (P8 bound 10%).
Over the monthly windows after its origin (2025-03, 2025-04, 2025-05, 2025-06, 2025-07, 2025-08), the origin-1 model's macro-F1 had Kendall tau = -0.20 against time (one-sided P for a decline = 0.3597; P9 predicts tau < 0); AUT = 42.6 points.

## Sensitivity to the near-duplicate threshold (deviation D1)

At the registered thresholds (tau_J = 0.535, 0.532, 0.528 at origins 1, 2, 3) the rule kept MELD-DS-448 167 of 1,001, 127 of 848, 157 of 649; WinMET 32 of 1,001, 34 of 1,018, 31 of 1,007 shared-family test traces at origins 1, 2, 3. H1 was inconclusive; H2 was inconclusive; H3 was inconclusive; H4 was not tested (unlike the primary analysis for H2, H3, H4); psi2 = 1.2 and 0.7 points, transfer gaps 14.5 and 16.6 points, rho = -0.105.
At tau_J = 0.7 the rule kept MELD-DS-448 418 of 1,001, 343 of 848, 313 of 649; WinMET 204 of 1,001, 208 of 1,018, 205 of 1,007 shared-family test traces at origins 1, 2, 3. H1 was inconclusive; H2 was inconclusive; H3 was supported; H4 was rejected (unlike the primary analysis for H2); psi2 = 0.1 and -0.8 points, transfer gaps 22.3 and 33.0 points, rho = 0.020.
At tau_J = 0.9 the rule kept MELD-DS-448 955 of 1,001, 816 of 848, 633 of 649; WinMET 838 of 1,001, 855 of 1,018, 845 of 1,007 shared-family test traces at origins 1, 2, 3. H1 was inconclusive; H2 was rejected; H3 was supported; H4 was rejected (as in the primary analysis); psi2 = -0.2 and 0.1 points, transfer gaps 22.2 and 42.2 points, rho = 0.048.
Supplementary Table S_tau lists every estimate with its 95% interval for each threshold.

## Discussion skeleton

[Authors: interpret the four verdicts above against S6's predictions, compare with Table 1, then state the threats to validity (single family taxonomy per installation, label noise, sequence cap, lexicon boundary choices) and the limitations recorded at S9.]
