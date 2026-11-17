# Methodology

## Proposed evidence framework

Supplementary Table S1 defines the notation used in this section. The framework treats a behavioural trace as an evidence item with three parts held apart: the token sequence, the environment record and the label record (Fig. 1a). A trace mixes three token classes. Behaviour (B) is what the malware would do in any environment. Footprint (F) is a value set by the environment itself, such as a user-profile path or a machine name. Probing (P) is a call by which the malware checks for an analysis environment, following the ATT&CK techniques for virtualisation and sandbox evasion (T1497). Reliance on F contaminates a finding; reliance on P is genuine but contingent on the environment.

Five layers each pass one named, hashed artefact to the next (Fig. 1b): acquisition, preservation in a hash-chained ledger whose batch roots are time-stamped by an external authority, feature engineering into class tokens, analysis by four learners, and explanation and reporting. Privacy controls and standard clauses apply to every layer. The workflow (Fig. 2a) quarantines any report whose digest does not match its source, seals it, tokenises it and only then scores it. An environment-match gate attaches the error rate validated under matching conditions, or refers the trace to an analyst. The ledger (Fig. 2b) records the frozen lexicon, hypotheses and split manifests before the first test score, so the order of events can be checked by anyone holding the ledger. Fig. 3 shows the analysis model with its masking operators and the evaluation protocol. In this study the freeze entry (ledger entry 130,774) was anchored by a trusted time-stamp on 5 November 2026, two days before the public registration and before any test trace was scored.

{{FIG:1}}

::: {custom-style="Figure Legend"}
**Figure 1. Behavioural malware evidence and the framework that handles it.** **a**, Conceptual model (diagram D1). A malware sample runs in an analysis environment and leaves an execution trace that mixes three token classes: behaviour (B) that the malware would show in any environment, probing (P) of the environment by the malware, and footprint (F) values that the environment itself sets. The trace is stored as an evidence item with its environment record *e*, supports an analytical finding and enters a forensic report. Dotted and dashed arrows mark the two threats to validity that the audit measures: reliance on P, which is genuine but environment-contingent, and reliance on F, which contaminates the finding. **b**, Layered architecture (diagram D2). Layers L1–L5 each pass one named artefact to the next; digests of models, results and findings return to the ledger in L2 (dashed arrow). The privacy and governance plane and the standards plane apply to every layer; clause numbers refer to ISO/IEC 27037:2012. Colour and line style carry the same information (B solid, P dotted, F dashed), so the figure reads in greyscale. 1D-CNN, one-dimensional convolutional neural network; API, application programming interface; ATT&CK, MITRE Adversarial Tactics, Techniques and Common Knowledge; BiLSTM, bidirectional long short-term memory network; BSA, Bharatiya Sakshya Adhiniyam 2023; CAPEv2, CAPE sandbox version 2; CSF, NIST Cybersecurity Framework 2.0; FRE, US Federal Rule of Evidence; HMAC, hash-based message authentication code; ISO/IEC, International Organization for Standardization and International Electrotechnical Commission; JSON, JavaScript Object Notation; NC and NC-SA, Creative Commons non-commercial and non-commercial share-alike licences; NIST, US National Institute of Standards and Technology; OS, operating system; RFC, Request for Comments; SHA-256, Secure Hash Algorithm with a 256-bit digest; SP, NIST Special Publication; T1055 and T1497, ATT&CK techniques for process injection and for virtualisation or sandbox evasion; WinMET, Windows Malware Execution Traces.
:::

{{FIG:2}}

::: {custom-style="Figure Legend"}
**Figure 2. Evidence workflow and chain of custody.** **a**, End-to-end workflow (diagram D3) in four swim lanes. A report whose SHA-256 digest does not match its source is quarantined and re-acquired. The environment-match gate decides which error rate accompanies a finding: a trace from a validated installation whose footprint attribution share φ~F~ under the base model stays at or below the threshold θ~F~ is reported automatically with the within-installation error rate; any other trace carries the cross-installation error rate and goes to analyst review. A new installation joins the validated set *E*~val~ only after re-validation (dashed loop). Italic labels give the standard clause or legal provision that each step implements. **b**, Hash-chained ledger and batch anchoring (diagram D4). Each entry stores the previous hash, so *h*~*i*~ = SHA-256(entry *i* ‖ *h*~*i*−1~); dashed links stand for omitted entries. The shaded entry *k* freezes the lexicon, hypotheses, margins and split manifests before the first test score (entry *k*+1), and the RFC 3161 token on the batch Merkle root *R*~b~ proves the order τ~*k*~ < τ~*k*+1~. In the tamper example, editing entry *k* after scoring makes every later hash, the root and its token fail verification. 800-86, NIST SP 800-86 (integrating forensic techniques into incident response); 27037, 27041 and 27042, ISO/IEC standards on evidence handling, on method validation, and on analysis and interpretation; BSA, Bharatiya Sakshya Adhiniyam 2023; CSF RS.AN, NIST Cybersecurity Framework 2.0 incident-analysis outcomes; FRE, US Federal Rule of Evidence; GDPR, EU General Data Protection Regulation; HMAC, hash-based message authentication code; ID, identifier; RFC 3161, Internet time-stamp protocol; SHA-256, Secure Hash Algorithm with a 256-bit digest; TST~b~, time-stamp token for batch *b*; H1–H4, the four pre-registered hypotheses.
:::

## Evidence integrity, standards and privacy

Every downloaded file and extracted record is an evidence item with its SHA-256, environment-record digest, provenance and custody fields. Ledger entries are chained by hashing each entry together with the previous hash (equation ({{EQREF:25}})); each batch of entries receives a Merkle tree hash in the form of RFC 6962 and RFC 9162, over which an RFC 3161 token is obtained (equation ({{EQREF:26}})):

{{EQ:25}}

{{EQ:26}}

An edit to any entry breaks every later hash and the token on its batch, so undetected tampering requires a hash collision (equation ({{EQREF:27}})), and a result is valid only if the freeze anchor precedes the first scoring entry (equation ({{EQREF:28}})):

{{EQ:27}}

{{EQ:28}}

The ledger held 130,758 entries after acquisition and sealing, and the chain, every anchored Merkle root and every token verified before the freeze. The results file is ledger entry 130,782. Because the first anchor of its batch was issued by the local development time-stamping authority, a re-anchor entry was appended and anchored by the external authority (entry 130,785); every entry's hash includes its predecessor, so this trusted token covers all earlier entries. Ledger actors are pseudonymised with a keyed hash (HMAC-SHA-256). Write-once storage was emulated with append-only files, read-only permissions and anchored roots.

The workflow maps to ISO/IEC 27037 for identification, collection, acquisition and preservation [@iso27037], ISO/IEC 27041 for method validation [@iso27041], ISO/IEC 27042 for analysis, interpretation and reporting [@iso27042], NIST Special Publication 800-86 for integrating forensic techniques into incident response, and the incident-analysis outcomes of NIST Special Publication 800-61 Revision 3 [@nist80061r3] (Fig. 2a). Models only ever see API names and argument classes, a residual-data scan for e-mail addresses, IP addresses, security identifiers, MAC addresses and user paths must pass before any release, and packed archives are deleted after hashing and extraction. These controls follow data minimisation and data protection by design under the EU General Data Protection Regulation (Articles 5 and 25) and the security-safeguard duties of India's Digital Personal Data Protection Act 2023, whose Rules were notified on 13 November 2025.

## Study design, registration and deviations

The study is a pre-registered computational experiment on public datasets. Hypotheses, estimands, margins, decision rules, family sets, split manifests and the token lexicon were frozen as files, written into the evidence ledger (entry 130,774) and anchored by a time-stamp token under the Internet time-stamp protocol (RFC 3161) from an external authority (freetsa.org) at 20:13:04 GMT on 5 November 2026. The analysis code refused to score test traces before this anchor. The registration was filed with OSF Registries from the same files and approved on 7 November 2026 (https://doi.org/10.17605/OSF.IO/K5EPU); it holds the hypotheses, decision rules and the SHA-256 digests of the frozen files. The token-level lexicon is embargoed and available to reviewers, so that lists of environment artefacts are not published as evasion aids.

The four hypotheses concern the confirmatory learner, a Transformer encoder, and are stated in points of family-balanced macro-F1 with a margin Δ of 3 points. H1: artefact tokens alone identify families, compared with trace length alone. H2: masking footprint and probing tokens at test time lowers macro-F1 more than matched random masks. H3: cross-installation macro-F1 on shared families is lower than within-installation macro-F1, in both directions. H4: training with footprint tokens masked recovers more than 10% of the gap without losing more than 3 points within either installation; H4 was tested only if H3 was supported. Each hypothesis ended as supported, rejected or inconclusive under the rules below.

Five amendments were made before the freeze and are recorded in the change log: streamed ingestion and stages to bound memory (amendments 1 and 2); a label check against an independent labeller, a revised dry run and a central-processing-unit (CPU) plan with a sequence cap of 256 tokens (3); corrected dry-run training and timing (4); and, because the dry run projected an interval half-width of 5.8 points against the planned 2 (kill criterion K5), a margin of 3 points with five seeds and 20 search configurations (5). One deviation followed the freeze. The registered rule set the near-duplicate threshold τ~J~ to the antimode of the nearest-neighbour Jaccard similarity density on each origin's training union. Before any model was trained or scored, it returned τ~J~ = 0.535, 0.532 and 0.528 and classed 76–85% of the MELD-DS-448 and 97% of the WinMET shared-family test traces as near-duplicates, leaving 31–34 WinMET test traces per origin; the similarity density on each training union had peaks near 0.47 and 0.89 and its minimum near 0.53 (Supplementary Fig. S1). We therefore used the registered fallback τ~J~ = 0.8 as the primary threshold and report the registered thresholds, 0.7 and 0.9 as sensitivity analyses of the same models. The decision (deviation D1) was recorded in the ledger (entry 130,778) with a trusted time-stamp before the first test trace was scored.

## Data sources

Installation I1 is MELD-DS-448 [@meld2025] (Hugging Face, commit e2dbf22; licence CC BY-NC-SA 4.0). We downloaded the dataset manifest and six archive volumes of CAPEv2 reports and API-call sequences, each matching its published SHA-256; disassembly and Markdown archives were skipped by design. Installation I2 is WinMET [@raducu2025dib] (Zenodo record 16414116, concept DOI 10.5281/zenodo.12647555; licence CC BY-NC 4.0): five archive volumes and three label files, each matching its published MD5 checksum. Both installations ran CAPEv2 on Windows 10 virtual machines. MALVADA (commit 2497402, licence GPL-3.0) provided the curation criteria [@raducu2025softx], which removed 22 MELD-DS-448 reports (21 without calls in the main process and one without processes) and no WinMET report. EMBER2024 Windows feature archives (Hugging Face revision 3d23efe; Apache-2.0) [@joyce2025] were read only for the family labels of SHA-256s shared with I1 or I2; no feature values were stored. No malware binary was downloaded, unpacked or executed. Every downloaded file is listed with its source, version, licence, size and digest in the Data Provenance Log (Supplementary Table S2).

A field audit found the guest-build field absent in both installations. MELD-DS-448 reports carry the CAPE version, machine, timeout, network route and analysis start; WinMET reports carry only the analysis start, so its environment record was taken from the WinMET publication. Median trace lengths were 3,124 calls in MELD-DS-448 and 6,729 in WinMET, and 5.5% and 0.5% of traces had fewer than 100 calls. Neither installation records VirusTotal scan dates.

## Tokens, sequences and lexicon

Each API call becomes one token (equation ({{EQREF:1}}); notation in Supplementary Table S1):

{{EQ:1}}

where a is the API name, κ(args) the tuple of argument classes (path category, registry hive and key category, IP-address flag, file-extension class, sleep-duration bucket and process-name class), r(ret) the return class, H~c~ the common hook set and ⟨OOS⟩ an out-of-set token. Raw strings, handles, addresses, process identifiers and time stamps have no slot in a token, so they never reach a model. The common hook set contains the 358 API names, of 467 observed, that occur in the training data of both installations. Calls from all processes are put in call-time order, consecutive repeats are collapsed with a run-length bucket, and sequences are cut head-plus-tail (equation ({{EQREF:2}})):

{{EQ:2}}

The cap L is the smallest power of two at or above the 95th percentile of collapsed training lengths, limited to 256 by the CPU plan. Token types seen in fewer than five training traces become an unknown token after their class has been assigned. The frozen lexicon assigns each of the 10,398 token types to behaviour (9,793 types), footprint (235) or probing (368). Probing rules derive from the ATT&CK sub-techniques of virtualisation and sandbox evasion (T1497) and from CAPEv2 documentation [@alsada2025]; footprint rules cover value classes set by the environment. The out-of-set token is classed F, and B in a sensitivity analysis. The lexicon was sealed in the ledger before any statistic comparing the installations was computed. As a pre-specified mechanism check, footprint tokens overlapped less between the installations than behaviour tokens (weighted Jaccard overlap 0.301 against 0.397; difference 0.095, 95% bootstrap confidence interval (CI) 0.061 to 0.125; Supplementary Equation S10).

## Masking audit and probes

Masking operators replace tokens of a class set S with a mask token, either at test time or at random during training with artefact dropout at rate p (equation ({{EQREF:3}})):

{{EQ:3}}

The control for H2 masks random positions matched to the footprint-and-probing mask in each cell of training-frequency band and position bin (equation ({{EQREF:4}})), using the finest grid that borrows at most 10% of positions on training traces:

{{EQ:4}}

Test conditions were no mask, footprint (M~F~), probing (M~P~), both (M~FP~), behaviour (M~B~, for the artefact-only probe) and the matched random mask. Three probes complemented the learners: a length-only probe (gradient-boosted trees on call count, distinct API names, processes, duration and an early-exit flag), an artefact-only probe that saw footprint and probing tokens with behaviour masked, and an environment-identification probe (logistic regression on frozen Transformer embeddings, assessed by permutation).

## Learners and training

The Transformer encoder has token, run-length and position embeddings, two to four layers, four heads, a model width of 128 and mean pooling, with masked scaled dot-product attention and a softmax over the K shared families (equations ({{EQREF:5}}) and ({{EQREF:6}})):

{{EQ:5}}

{{EQ:6}}

The comparison learners were a bidirectional long short-term memory network (BiLSTM) with attention pooling, built on long short-term memory units [@hochreiter1997], a one-dimensional convolutional neural network (1D-CNN) with kernel widths 3, 5 and 7 and max pooling, and gradient-boosted trees [@friedman2001] (LightGBM 4.6.0) on term frequency–inverse document frequency weights of token 1–3-grams (n-gram LightGBM; Supplementary Equations S1–S3). Deep learners minimised class-weighted cross-entropy under artefact dropout (equation ({{EQREF:7}})):

{{EQ:7}}

with γ ∈ {0, 0.5, 1}. Each learner except the BiLSTM received 20 random-search configurations drawn from a fixed seed; the BiLSTM used one fixed configuration under the reduced compute plan. All learners were trained with five seeds (11, 23, 37, 41 and 53), the neural learners for at most 20 epochs with early stopping after three epochs without improvement on the validation month. The artefact-dropout rate p ∈ {0.25, 0.5, 0.75, 1} was chosen on validation-month macro-F1 with footprint tokens masked within the installation, so tuning never saw the other installation. Under the CPU plan the BiLSTM was trained unmasked and footprint-masked only, and the n-gram LightGBM ran the full design. Search spaces and selected settings are given in Supplementary Table S3, and pseudocode for sealing, freezing, sequence building, the audit and casework scoring is given in the Supplementary Information.

## Splits, families and near-duplicate control

MELD-DS-448 was split at three rolling origins: validation months February, April and June 2025, test windows March–April, May–June and July–August 2025, and training on all earlier months. WinMET has no submission dates, so it was split once into training, validation and test parts that keep near-duplicate clusters together. Family labels came from the MELD-DS-448 card and from AVClass for WinMET, harmonised by a 15-entry alias table that the authors confirmed before the freeze. The label check compared each installation with ClarAVy family labels in EMBER2024 [@joyce2025] for shared SHA-256s with a family confidence of at least 0.7; a family was dropped if fewer than 80% of at least ten such binaries agreed. Agreement was 54.5% of 356 confident shared binaries for MELD-DS-448 and 93.2% of 409 for WinMET. Two families were dropped (agenttesla, 72.7% of 11; formbook, 15.4% of 13), two were verified (amadey and quasar) and seven had too few binaries to judge and were kept as unverified. A family was analysed at an origin only with at least 20 training-period and 10 test traces in both installations, giving amadey, gcleaner, guloader, quasar, redline, remcos and snakelogger at every origin, loki at origins 1–2 and nanocore at origins 2–3.

Near-duplicates were defined on sets of API-name 3-grams with the Jaccard index, estimated by MinHash [@broder2000] with locality-sensitive hashing to propose candidate pairs; decisions used the exact index (equation ({{EQREF:8}})):

{{EQ:8}}

A test trace left the analysed set when its largest Jaccard index to any training or validation trace of its origin reached τ~J~, and near-duplicate clusters within each test set defined the unit of resampling (equation ({{EQREF:9}})):

{{EQ:9}}

A same-binary rule removed test traces whose SHA-256 had a report in the training set; it removed none. The paired same-binary design, which would have compared identical binaries across installations, needed 200 shared binaries in five families and did not qualify (41 binaries in ten families).

## Estimands and statistical analysis

Figure {{F:design}} shows how the models, probes and test conditions combine into the estimands of the four hypotheses. Per-family F1 and macro-F1 are defined in equation ({{EQREF:11}}); the primary metric is the family-balanced variant of equation ({{EQREF:12}}), which removes the effect of different family priors on precision:

{{EQ:11}}

{{EQ:12}}

The transfer gap (equation ({{EQREF:13}})), the recovered share of the pooled gap (equation ({{EQREF:14}})) and the effects ψ~1~–ψ~4~ (equation ({{EQREF:15}})) operationalise H1–H4:

{{EQ:13}}

{{EQ:14}}

{{EQ:15}}

Macro-F1 was averaged over the five seeds and, for MELD-DS-448, over the three origins. Uncertainty came from a cluster bootstrap [@field2007] with 9,999 replicates that resamples near-duplicate clusters within test set, origin and family and recomputes every estimate (equation ({{EQREF:16}})); percentile intervals and one-sided bootstrap P values (equation ({{EQREF:17}})) need no normality assumption and respect the dependence between near-duplicate traces:

{{EQ:16}}

{{EQ:17}}

Claims about both installations or both directions used the intersection–union test [@berger1982] (equation ({{EQREF:18}})), and Holm's step-down procedure adjusted the four hypotheses, separately for support and for rejection (equation ({{EQREF:19}})):

{{EQ:18}}

{{EQ:19}}

A hypothesis was supported when its adjusted support P value was at most α/2 = 0.025 (one-sided), rejected when it was not supported and its adjusted rejection P value was at most 0.025, and inconclusive otherwise. H2 was rejected by two one-sided tests of equivalence within ±Δ [@schuirmann1987] (equation ({{EQREF:20}})):

{{EQ:20}}

The sample sizes are the analysed test traces given in the Results (1,802 MELD-DS-448 and 975 WinMET traces at τ~J~ = 0.8). Exploratory analyses, labelled as such, were an exact McNemar test on seed-majority predictions, a Wilcoxon signed-rank test with the Hodges–Lehmann estimate on family-level F1 differences, Kendall's τ with an exact one-sided test for the monthly trend of the origin-1 model, summarised by area under time [@pendlebury2019], and permutation tests for the environment-identification probe (Supplementary Equations S7–S9). The design effect of the test clusters was reported against the registered expectation of at most 1.2.

{{FIG:3}}

::: {custom-style="Figure Legend"}
**Figure 3. Analysis model, masking audit and evaluation protocol.** **a**, Analysis model (diagram D5). Each CAPEv2 report is reduced to class tokens *t* = (*a*, κ, *r*), that is API name, argument classes and return class, over a hook set *H*~c~ common to both installations. The frozen lexicon tags each token B, P or F, and masking operators replace F tokens, P tokens, both, or a matched random set with a mask token (M). Four learners output a family posterior *f*~θ~(*x*) over *K* shared families; Integrated Gradients (IG) gives the attribution shares φ~B~, φ~P~ and φ~F~, and the environment record *e* selects the matching error rate. The design crosses masking at training with masking at test; the mitigated model trains with F masked by artefact dropout at rate *p*. **b**, Evaluation protocol (diagram D6). MELD-DS-448 (I1) is evaluated at three rolling origins with non-overlapping two-month test windows in 2025, each preceded by a validation month; WinMET (I2) has no submission dates and uses one split grouped by near-duplicate cluster. Five harmonisation steps precede scoring, and transfer runs in both directions on shared families. The paired same-binary test did not qualify in this study (41 shared binaries). The statistics and the three-way decision rule were fixed before scoring, and gates K3–K6 could stop or change the protocol. 1D-CNN, one-dimensional convolutional neural network; 22H2, Windows 10 version 22H2; BiLSTM, bidirectional long short-term memory network; CAPEv2, CAPE sandbox version 2; CI, confidence interval; F1, harmonic mean of precision and recall; K3–K6, pre-registered kill criteria; LightGBM, light gradient-boosting machine; MinHash, min-wise hashing; TF-IDF, term frequency–inverse document frequency; TOST, two one-sided tests; Δ, smallest effect of interest (3 macro-F1 points).
:::

{{FIG:design}}

::: {custom-style="Figure Legend"}
**Figure {{F:design}}. Design of the masking audit and the four estimands.** Models trained on one installation (left) are scored on test traces from the same or the other installation under test-time masks (middle), and the estimands of hypotheses H1–H4 compare these scores (right). *f*^len^, probe that sees trace length only; *f*^art^, probe that sees footprint and probing tokens only, with behaviour tokens masked; *f*^base^, base model on all tokens; *f*^mit^, model trained with footprint tokens masked and artefact dropout. *M*~F~, *M*~P~, *M*~FP~ and *M*~B~ mask footprint, probing, footprint-and-probing and behaviour tokens; *M*~rand~ masks random tokens matched to *M*~FP~ in token frequency and position. F1, family-balanced macro-F1 averaged over five training seeds and, for MELD-DS-448, three origins. Margins are 3 points for ψ~1~–ψ~3~ and ψ~4b~ and 0.10 for ρ; one-sided tests at 0.025 per claim, combined across installations by the intersection–union rule and Holm-adjusted over H1–H4.
:::

## Explanations and referral

Integrated Gradients [@sundararajan2017] attributed the logit of the predicted family to token positions against a baseline of mask tokens, using Gauss–Legendre quadrature with m = 50 nodes, raised to 100 and 200 if the completeness criterion failed (equations ({{EQREF:21}}) and ({{EQREF:22}})):

{{EQ:21}}

{{EQ:22}}

Absolute attribution was summed by token class (equation ({{EQREF:23}})) for 200 explained test traces per cell. Pre-specified checks required zero attribution at masked positions and a relative completeness gap δ(x) of at most 0.05 for at least 95% of explained traces.

{{EQ:23}}

The referral rule of the environment-match gate (equation ({{EQREF:24}})) refers a trace whose environment record does not match a validated installation, or whose footprint share under the base model exceeds the 95th percentile θ~F~ of validation traces (budget η~F~ = 0.05):

{{EQ:24}}

## Audit algorithm

Algorithm 1 summarises the audit as numbered steps; the Supplementary Note on pseudocode gives each stage in full.

**Algorithm 1. Evidence-sealed audit of family attribution across sandbox installations.** Input: CAPEv2 reports from two installations, the frozen lexicon and the registered hypotheses H1–H4. Output: sealed findings with validated error rates, explanations and ledger proofs.

1. For every source file, compute its SHA-256 digest and compare it with the published digest; quarantine the file on a mismatch, otherwise append a hash-chained ledger entry (equation ({{EQREF:25}})).
2. At the end of each batch, compute the Merkle root of its entries and anchor it with an RFC 3161 time-stamp token (equation ({{EQREF:26}})).
3. Before any test trace is scored, freeze the lexicon, hypotheses, margins, family thresholds and split manifests, seal them in the ledger and anchor the freeze entry; accept a test score only if the anchor precedes the first scoring entry (equation ({{EQREF:28}})).
4. Build harmonised token sequences: map each call to its API name, argument classes and return class, restrict calls to the common hook set, interleave processes by call time, collapse repeats with a run-length bucket, cut each trace to 256 tokens and tag every token as behaviour, footprint or probing (equations ({{EQREF:1}}) and ({{EQREF:2}})).
5. Form the rolling-origin splits of MELD-DS-448 and the grouped split of WinMET, and remove test traces that are near-duplicates of training traces or whose binary has a report in the training set (equations ({{EQREF:8}}) and ({{EQREF:9}})).
6. For each learner, seed and origin, train a base model on unmasked tokens and a mitigated model with footprint tokens masked and artefact dropout, together with the length-only, artefact-only and environment-identification probes.
7. Score every model within and across installations under no mask, footprint, probing, footprint-and-probing and frequency-matched random masks.
8. Compute family-balanced macro-F1, the estimands of H1–H4, 95% cluster-bootstrap intervals and one-sided P values, combine installations by the intersection–union rule, apply the Holm adjustment and seal every number in the ledger.
9. For the explained test traces, compute Integrated Gradients shares of behaviour, footprint and probing tokens and refer a trace to an analyst when its footprint share exceeds the 95th percentile of validation traces.
10. Report each finding with the error rate validated under a matching environment, its ledger entry, Merkle path and time-stamp token, or refer the trace to an analyst when the environment does not match.

## Use of large language models

The author used Claude chat and Claude Cowork assistants (Anthropic; models including Claude Opus 5.5) between 31 October and 17 November 2026. They were used for literature discovery and bibliographic verification support, drafting and testing analysis code, drafting the diagram code, and drafting and language editing of the manuscript. The author ran every analysis on his own computer, checked every reported number against the results file recorded in the ledger, verified every reference at its source, and takes responsibility for the content. No image was generated by an AI model; all figures were drawn from code.

## Ethics and dual use

The study involved no human participants, human data or animals. The author did not download, unpack or execute any malware binary; the data are public sandbox reports and labels used under their licences. Released derived data contain class tokens only, without raw argument values or network indicators, and the token-level lexicon of environment artefacts is shared with reviewers rather than published, so the work does not provide a recipe for evading sandboxes.

# Experimental setup

## Datasets and test traces analysed

We compared family attribution from CAPEv2 API-call traces in the two installations on the malware families both share: 8, 9 and 8 families at three rolling origins whose MELD-DS-448 test windows were March–April, May–June and July–August 2025. WinMET has no submission dates, so it was split once into groups of near-duplicate traces and analysed with each origin's family set (Methodology). The confirmatory learner was a Transformer encoder. The BiLSTM, the 1D-CNN and the n-gram LightGBM were exploratory.

The primary analysis kept a test trace when its largest Jaccard index to the training traces of its origin was below 0.8 and its binary had no report in that training set. This threshold is deviation D1 from the registered antimode rule; it was recorded and time-stamped before any test trace was scored (Methodology). The rule kept 705 of 1,001, 589 of 848 and 508 of 649 MELD-DS-448 shared-family test traces at origins 1–3, and 322 of 1,001, 329 of 1,018 and 324 of 1,007 WinMET traces. The registered thresholds (0.535, 0.532 and 0.528) would have kept 127–167 MELD-DS-448 and 31–34 WinMET traces per origin. All analyses use the same trained models.

Figure {{F:flow}} traces the evidence from the released reports to the analysed test traces, and Table {{T:data}} gives the split sizes at each origin. Of 26,166 MELD-DS-448 reports, 26,144 met the curation criteria, and all 31,844 WinMET traces did. The MELD-DS-448 training sets grew from 1,004 to 2,821 traces across the rolling origins, whereas the WinMET training sets held 4,814–4,857 traces of the same families. The near-duplicate rule kept 69.5–78.3% of the MELD-DS-448 test traces but only 32.2–32.3% of the WinMET test traces: about two thirds of the WinMET test traces had a near-identical trace in training. Finding these pairs took 21.8–28.3 million Jaccard comparisons per origin.

{{FIG:flow}}

::: {custom-style="Figure Legend"}
**Figure {{F:flow}}. Evidence flow and sample sizes.** Counts of reports and traces from release to analysis in each installation and at each rolling origin. Training, validation and test counts are given before near-duplicate control; kept, test traces whose largest Jaccard index to the training union of their origin is below 0.8 and whose binary has no report in the training set. Families are those present in both installations at each origin. CAPEv2, CAPE sandbox version 2; MALVADA, the curation criteria of Raducu et al.; MD5 and SHA-256, published checksums verified at download. Source: assembly record of the evaluation.
:::

**Table {{T:data}}. Training, validation and test traces per origin, and the near-duplicate search.**

| Origin | Families | MELD-DS-448 test window | MELD-DS-448 training / validation / test (kept) | WinMET training / validation / test (kept) | Vocabulary, MELD-DS-448 / WinMET models | Traces searched | Pairs compared (millions) |
|:--------|:--------|:----------|:---------------|:---------------|:-----------|:----------|:----------|
| 1 | 8 | March–April 2025 | 1,004 / 366 / 1,001 (705) | 4,814 / 379 / 1,001 (322) | 1,658 / 1,486 | 33,791 | 21.8 |
| 2 | 9 | May–June 2025 | 1,936 / 456 / 848 (589) | 4,857 / 384 / 1,018 (329) | 1,895 / 1,500 | 40,180 | 24.7 |
| 3 | 8 | July–August 2025 | 2,821 / 325 / 649 (508) | 4,821 / 379 / 1,007 (324) | 2,051 / 1,493 | 45,747 | 28.3 |
| Total kept | | | 1,802 | 975 | | | |

Families, families present in both installations at the origin. Kept, test traces retained by the near-duplicate rule at the primary threshold (largest Jaccard index to the training union below 0.8, and the binary has no report in the training set). Vocabulary, distinct tokens seen in training by models trained on each installation; every sequence was cut to 256 tokens. Traces searched, traces in the training union of the origin; pairs compared, Jaccard comparisons made by the near-duplicate search. Source: assembly record of the evaluation (assembly_info.json).

## Evaluation metrics and statistical reporting

Estimates are family-balanced macro-F1 in points, averaged over five training seeds and the three origins, with 95% cluster-bootstrap percentile CIs from 9,999 replicates. P values are one-sided, combined across installations by the intersection–union rule and Holm-adjusted over the four hypotheses at a one-sided level of 0.025 per claim.

## Software and computation

All models were trained and scored on the author's personal computer without a graphics processor, under Windows Subsystem for Linux (Ubuntu 24.04) with 12 logical CPUs, Python 3.12.16, PyTorch 2.5.1 (CPU build), NumPy 2.1.3, SciPy 1.14.1, scikit-learn 1.5.2, pandas 2.2.3, LightGBM 4.6.0, ijson 3.3.0, py7zr 0.22.0 and Matplotlib 3.9.2. The dry run projected 58.3 CPU hours for the registered plan with eight threads; the evaluation stage took 57.1 h of wall-clock time (205,570 s), and test-set assembly, including the near-duplicate search, took 1.25 h with a peak memory of 1.3 GB (Table {{T:cost}}). The harness and evaluation engine passed their unit tests (40 and 50 tests) before use, and the frozen code is identified by the SHA-256 digests in the code release. Search spaces, settings and the computing environment are listed in Supplementary Table S3.

**Table {{T:cost}}. Computational cost of the audit on a personal computer without a graphics processor.**

| Stage or learner | Random-search configurations | Mean epochs, base model | Mean epochs, F-masked model | Summed job time (h) |
|:----------------|:--------------|:--------|:--------|:--------|
| Transformer | 20 per fit group | 10.6 | 10.3 | 158.2 |
| BiLSTM | None (reduced plan) | 10.1 | 10.0 | 214.4 |
| 1D-CNN | 20 per fit group | 8.4 | 8.9 | 16.5 |
| n-gram LightGBM | 20 per fit group | — | — | 3.2 |
| Near-duplicate search and assembly | — | — | — | 1.25 (wall clock) |
| Whole evaluation stage | — | — | — | 57.1 (wall clock) |

Each learner had six fit groups (three origins in each installation); deep learners were trained with five seeds for base and F-masked models, for at most 20 epochs with early stopping. Summed job time adds the time of all jobs of a learner and exceeds the wall-clock time because jobs ran in parallel. Assembly compared 74.8 million trace pairs with a peak memory of 1.3 GB. Source: fit and assembly records of the evaluation.
