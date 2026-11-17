# OSF registration draft: sandbox-artefact audit of deep-learning malware family attribution (A15)

**Filed and approved on 7 November 2026: DOI [10.17605/OSF.IO/K5EPU](https://doi.org/10.17605/OSF.IO/K5EPU); associated project https://osf.io/f5gtb; Internet Archive copy https://archive.org/details/osf-registrations-k5epu-v1. The registered text is the public record; this draft is kept for the study files.**

*Draft for the authors, 5 November 2026 (freeze anchor filled in at 20:13 UTC). Edit before filing. The headings follow the OSF Preregistration template, so each section can be pasted into the matching field. The frozen files in `WORK\outputs\freeze_bundle\` are authoritative: if this text and a frozen file disagree, the file wins. Placeholders are in [square brackets].*

## Study information

**Title.** Sandbox-artefact audit of deep-learning malware family attribution (A15)

**Authors.** [Names, affiliations, ORCID iDs]

**Description.** Deep-learning models that attribute Windows malware to families from sandbox API-call traces may learn the sandbox rather than the malware: tokens that reflect the analysis environment (environment footprint, F) or the malware's checks for an analysis environment (probing, P), rather than the rest of its behaviour (B). We test this on two independent CAPEv2 installations, MELD-DS-448 (I1) and WinMET (I2), restricted to the malware families both share. A frozen lexicon assigns every token type to B, F or P before any test trace is scored. We measure whether artefact tokens alone identify families, whether masking F and P at test time costs more than matched random masks, how large the cross-installation attribution gap is, and how much of that gap F-masked training recovers.

**Hypotheses.** All differences are in points of family-balanced macro-F1. The margin Δ is 3 points (see "Other": amendment 5). Estimands and equation numbers refer to `hypotheses.json` and the study's mathematical model.

| | Claim | Estimand | Supported if | Rejected if |
| --- | --- | --- | --- | --- |
| H1 | Artefact tokens alone identify families | ψ1(I) = F1(f_art; M_B(D_te)) − F1(f_len; D_te), each installation | the lower bound exceeds 3 points in every installation (intersection-union test) | one installation's bound is below 3 |
| H2 | Test-time masking of F and P lowers macro-F1 more than matched random masks | ψ2(I) = F1(f_base; M_rand(D_te)) − F1(f_base; M_FP(D_te)), each installation | the lower bound exceeds 3 points in every installation | equivalence within ±3 points (two one-sided tests) in both installations |
| H3 | Cross-installation macro-F1 on shared families is lower than within-installation | ψ3(I→I′) = G(I→I′) (equation 13), both directions | the lower bound exceeds 3 points in both directions | the upper bound is below 3 in either direction |
| H4 | F-masked training recovers part of the gap without losing more than 3 points within either installation | ψ4a = ρ, the recovered share (equation 14, pooled over both directions); ψ4b(I) = F1(f_mit; M_F(D_te)) − F1(f_base; D_te) | ρ > 0.10 and ψ4b > −3 in every installation | ρ < 0.10 |

H4 is tested only if H3 is supported. f_base is the model trained on unmasked traces, f_mit the model trained with F-masking or artefact dropout, f_art a probe that sees artefact tokens only, f_len a probe that sees trace length only; M_B, M_F, M_FP and M_rand are test-time masks of B, F, F and P together, and matched random positions.

## Design plan

**Study type.** Other: pre-registered computational experiment on existing public datasets.

**Blinding.** No model has been scored on a test window. The analysis code refuses to score test traces until the frozen files are written into the study's hash-chained ledger and anchored by an RFC 3161 trusted time-stamp ("freeze"). The freeze was made on 5 November 2026: freeze entry 130,774; anchor entry 130,775 (Merkle root edf4b6f3f3bdcb105f06afdd75083d0d62fd0ad9ea1fd91227e87cebf388ac2b); time-stamp 5 November 2026, 20:13:04 GMT, from freetsa.org; the ledger verified in full.

**Study design.**
- Installations: I1 MELD-DS-448 (CAPEv2 reports, first seen April 2020 – August 2025); I2 WinMET (CAPEv2 reports, no VirusTotal dates).
- I1 uses three rolling origins: validation February, April and June 2025, with test windows March–April, May–June and July–August 2025. I2 uses one grouped split that keeps near-duplicate clusters together, because WinMET has no submission dates.
- Shared families per origin (after name harmonisation, the authors' alias table and the K4 label check): 8 at origin 1, 9 at origin 2 and 8 at origin 3. They are amadey, gcleaner, guloader, quasar, redline, remcos and snakelogger at every origin, loki at origins 1–2 and nanocore at origins 2–3.
- Training regimes: unmasked (f_base) and F-masked or artefact dropout (f_mit), with p ∈ {0.25, 0.5, 0.75, 1} chosen on validation-month macro-F1 with F masked within the installation. Test conditions: unmasked, M_F, M_FP, M_B and matched random masks. Probes: f_art and f_len. Transfer in both directions (I1→I2, I2→I1).
- Learners: Transformer encoder (confirmatory); BiLSTM with attention, 1D-CNN and n-gram LightGBM (secondary).
- Five training seeds (11, 23, 37, 41, 53), averaged; 20 random-search configurations per learner, with class-weight exponent γ ∈ {0, 0.5, 1}. The BiLSTM is trained unmasked and F-masked only (compute plan).

**Randomisation.** Training seeds are fixed as above, and the random-search configurations are drawn from a fixed seed.

## Sampling plan

**Existing data.** Registration prior to analysis of the data. The datasets are public and were acquired, sealed and audited before this registration. Without training or scoring any model on a test window, the following were computed: field completeness, family counts and split sizes, near-duplicate clusters, label agreement with an independent labeller (K4), and a token-overlap prediction (P3, no model involved). Models were trained only in a dry run on a pseudo-origin inside every origin's training period (I1 traces up to December 2024 for training, January–February 2025 for testing) and on WinMET's training and validation parts.

**Data collection procedures.** MELD-DS-448 (Hugging Face, commit e2dbf22; CC BY-NC-SA 4.0) and WinMET (Zenodo; CC BY-NC 4.0) were downloaded with published-digest checks. Only CAPEv2 reports and API-call sequences were read; no malware binary was downloaded, unpacked or executed. Every file and record is hashed into the ledger; versions, licences and digests are in the Data Provenance Log. EMBER2024 Windows feature archives (Hugging Face revision 3d23efef7c0f0b702c5024400cfff4c3744a3832; Apache-2.0) were read only for the labels of shared SHA-256s (K4).

**Sample size.** Test traces in the shared families: I1 1,001, 848 and 649 at origins 1–3; I2 1,001, 1,018 and 1,007 for the families of each origin. Training plus validation traces: I1 1,370, 2,392 and 3,146; I2 5,241.

**Sample size rationale.** The sizes are fixed by the data. The dry run projected a 95% half-width of 5.8 points for the cross-installation gap at the smallest origin, against the 2 points planned (kill criterion K5). The test windows cannot widen, because MELD-DS-448 ends in August 2025. As registered, the margin was raised to 3 points. The projected half-width still exceeds 3, so equivalence and non-inferiority verdicts may be inconclusive.

**Stopping rule.** Not applicable: every analysis runs once on the fixed data.

## Variables

**Manipulated variables.** Training regime (unmasked, F-masked or artefact dropout); test-time mask (none, M_F, M_FP, M_B, matched random); probe input (artefact tokens only, length only); transfer direction; learner.

**Measured variables.** Family-balanced macro-F1 (primary) and raw macro-F1; per-family recall; Integrated Gradients attributions per token class (logit target, ⟨M⟩ baseline, Gauss–Legendre quadrature with 50 nodes, raised to 100 and 200 if the completeness criterion fails); environment-identification probe accuracy; the referral statistic φ_F of the base model (budget η_F = 0.05).

**Indices.** ψ1–ψ4 as defined under Hypotheses; ρ pooled over both directions (per-direction values secondary).

## Analysis plan

**Statistical models.** Macro-F1 is averaged over the five seeds and, for I1, over the three origins. Uncertainty comes from a cluster bootstrap with 9,999 replicates (clusters: near-duplicate clusters within test set, origin and family) and percentile bounds. Support uses intersection-union tests over the components of each hypothesis. The rejection rule for H2 uses two one-sided tests (TOST) at ±3 points.

**Transformations.** Traces are tokenised by the frozen class tokeniser (closed enumerations only), consecutive repeats are collapsed with a run-length bucket min(⌊log₂ n_run⌋, 7), and sequences are cut head-plus-tail at L, the smallest power of two at or above the 95th percentile of training lengths, capped at 256. Token types seen in fewer than 5 training traces become ⟨UNK⟩. The out-of-set token is classed F, and B in a sensitivity analysis.

**Inference criteria.** One-sided α = 0.025 per claim. Holm correction over the four hypotheses, separately for the support family and the rejection family. Each hypothesis is declared supported, rejected or inconclusive.

**Data exclusion.**
- Traces failing MALVADA's curation criteria are excluded (22 I1 traces: 21 without calls in the main process, 1 without processes; no I2 traces).
- Catch-all family labels are excluded.
- A family is analysed at an origin only with at least 20 training-period and 10 test traces in both installations.
- Families whose labels disagree with ClarAVy (EMBER2024) on more than 20% of at least 10 shared binaries with confidence ≥ 0.7 are dropped: agenttesla and formbook. Seven families had too few such binaries to judge and are kept as unverified.
- Test traces with a near-duplicate in training are removed. The Jaccard threshold on API-name 3-grams is set by the antimode rule on training traces (fallback 0.8), MinHash with LSH only proposes candidate pairs, and decisions use exact Jaccard.

**Missing data.** The guest-build field is absent in both installations. I2's environment record is taken from the WinMET publication, because its reports carry only the analysis start. Traces without a family label are not used.

**Exploratory analysis.** The secondary learners, per-direction recovered shares, the out-of-set-token sensitivity analysis, the environment-identification probe and any analysis not defined in `hypotheses.json` are exploratory and will be labelled so.

## Other

**Kill criteria at the dry run.**
- K2 not triggered (both installations obtained).
- K3 not triggered (8, 9 and 8 shared families; minimum 5).
- K4 triggered: two families dropped, and the limitation is stated.
- K5 triggered: margin raised to 3 points.
- K6 triggered: CPU plan, about 58 h projected.

**Pre-registration amendments, all made before the freeze.**
1. Ingestion memory (3 November).
2. Streaming stages (3 November).
3. Label check, dry-run design and CPU plan with sequence cap 256 (4 November).
4. Dry-run training length and timing corrections (5 November).
5. Margin 3 points; five seeds and 20 search configurations restored (5 November).

Full notes are in the study's change log.

**Frozen files.** The token-level lexicon is embargoed and shared with reviewers; only its digest is public.

| File | SHA-256 |
| --- | --- |
| alias_table.json | 0a1897a800ebee03191de2599aaee67b75e404d5601161afff1b781a27dba0f9 |
| family_sets.json | f7216a24a3858235957ccf7594d6ac861a2f3da7ed0b3c201d8991d42244be12 |
| hypotheses.json | f1230869900b0ca396def191bce5374915f0fe405e95653a74bb2a64bfa51349 |
| lexicon.json | 6b5a02cfcc686218abf044e0ac54c32541e34dec8f67373b96e1dd71a89a8dec |
| settings.json | d1e2d75e144d81631e48d3ada58f99ddd86371540e932870146d99305e5cd8e5 |
| split_I1_o1.json | dcaa0785f5d159b3754302fbe71ddecb745f980566688a6f485d514a8f2a5663 |
| split_I1_o2.json | 69450c8c501d88888cba90bfb95f6c1b99713d61d7b5470aaddfeeccf8ade0ee |
| split_I1_o3.json | 96913cfcaa064473b2da9b978e4aaa5b0200f5917bae35c86a0f5a181e27c2bd |
| split_I2_grouped.json | 69806d58321ee14d4c32f3e47913432ef04475700dd6a6fded589fca25d10e3c |

Any change after this registration is reported as a deviation.
