::: {custom-style="Title"}
Deep learning attribution of malware families from sandbox behaviour traces transfers poorly between installations of the same sandbox
:::

Rijvan Beg^1,\*^

^1^Department of Computer Science and Engineering, SRM University-AP, Amaravati 522240, Andhra Pradesh, India

^\*^Corresponding author: Rijvan Beg (rijvan.beg@gmail.com)

# Abstract

Machine-learning models turn sandbox records of malware behaviour into findings such as the family a sample belongs to, and forensic use requires that the error rate of such a finding be known for the conditions under which the evidence was collected. We designed a framework that keeps each behavioural trace with a record of its analysis environment, seals every artefact, model and result in a hash-chained, externally time-stamped ledger, and audits learned attributions within and between sandbox installations. In a pre-registered study of two independent installations of the same open-source sandbox, sharing eight or nine malware families, a Transformer encoder attributed families with balanced F1 scores of 47.7 and 62.0 points (scale 0–100) within each installation but 23.3 and 24.1 points on the other installation's traces: gaps of 24.4 (95% confidence interval 20.5–27.6) and 38.0 (22.6–42.2) points. Hiding tokens that record the analysis environment, or the malware's checks for it, cost no more than hiding random tokens, and training without environment tokens recovered 4.6% of the gap (1.4–8.4%). Every learner lost 21–53 points between installations. A family attribution learned from sandbox traces is evidence only for the installation on which its error rate was measured.

**Keywords:** malware family attribution; sandbox behaviour traces; deep learning; digital forensics; evidence integrity; pre-registration

# Abbreviations

| Abbreviation | Definition |
|:-----------|:----------------------------------------|
| 1D-CNN | One-dimensional convolutional neural network |
| AI | Artificial intelligence |
| API | Application programming interface |
| ATT&CK | MITRE Adversarial Tactics, Techniques and Common Knowledge |
| B, F, P | Behaviour, footprint and probing token classes |
| BiLSTM | Bidirectional long short-term memory network |
| BSA | Bharatiya Sakshya Adhiniyam 2023 |
| CAPEv2 | CAPE sandbox, version 2 |
| CI | Confidence interval |
| CNN | Convolutional neural network |
| CPU | Central processing unit |
| CSF | NIST Cybersecurity Framework 2.0 |
| DLL | Dynamic-link library |
| FRE | United States Federal Rule of Evidence |
| GDPR | General Data Protection Regulation |
| GRU | Gated recurrent unit |
| H1–H4 | The four registered hypotheses |
| HMAC | Hash-based message authentication code |
| IG | Integrated Gradients |
| ISO/IEC | International Organization for Standardization and International Electrotechnical Commission |
| JSON | JavaScript Object Notation |
| LightGBM | Light gradient-boosting machine |
| LIME | Local interpretable model-agnostic explanations |
| LRP | Layer-wise relevance propagation |
| LSTM | Long short-term memory |
| macro-F1 | Family-balanced macro-averaged F1 score |
| MAML | Model-agnostic meta-learning |
| MinHash | Min-wise hashing |
| NIST | United States National Institute of Standards and Technology |
| NR | Not reported |
| OSF | Open Science Framework |
| PE | Portable executable |
| RFC | Request for Comments |
| ROC-AUC | Area under the receiver operating characteristic curve |
| SHA-256 | Secure Hash Algorithm with a 256-bit digest |
| SHAP | Shapley additive explanations |
| SP | NIST Special Publication |
| TF-IDF | Term frequency–inverse document frequency |
| TLSH | Trend Micro locality-sensitive hash |
| TOST | Two one-sided tests |

# Introduction

Malware analysis increasingly ends in a finding that someone must act on or defend: that a file is malicious, which family it belongs to, or what it did on a machine. Behavioural evidence underpins many such findings. A sandbox executes a sample in an instrumented virtual machine and records its application programming interface (API) calls, and the resulting traces support detection, family attribution and the reconstruction of malware activity [@maniriho2024jss; @maniriho2024csur]. Deep learning has become a common way of turning these traces into findings. Recurrent, convolutional and Transformer models learn directly from API-call sequences, with or without call parameters [@maniriho2023; @chen2022; @trizna2024], and new architectures target previously unseen malware [@joshi2025]. Digital forensics has begun to adopt the same tools for triage, attribution and incident response [@dunsin2024; @beg2026a; @beg2026b].

A forensic finding carries an obligation that a benchmark score does not. Under the 2023 amendment of United States Federal Rule of Evidence 702, an expert's opinion must reflect a reliable application of the principles and methods to the facts of the case, and international guidance on digital evidence asks that examination methods be validated for the conditions in which they are used [@iso27037; @iso27041; @iso27042]. For a learned model, this means that its error rate must be known for evidence collected under the same conditions as the evidence being examined. Interpretable and explainable models have been proposed to make forensic artificial intelligence (AI) more trustworthy [@solanke2022], and recent work turns learned decisions into signatures that analysts can reuse or tracks how forensic artefacts drift during long investigations [@beg2026a; @patidar2026]. An explanation, however, does not show that an error rate measured on evidence from one collection environment holds for evidence from another.

Admissibility tests make the same demand in other words. The Daubert criteria applied by United States courts ask, among other things, whether a technique has a known or potential error rate, and the amended Rule 702 requires the court to find it more likely than not that an expert's testimony meets the rule's conditions before it is admitted. India's Bharatiya Sakshya Adhiniyam 2023 requires a certificate for electronic records that identifies each record by its hash value, which places the integrity of digital evidence on the record. Neither prescribes a method, but together they make two questions unavoidable for any learned finding: on which data was its error rate measured, and can the record show that the analysis was fixed before the outcome was known?

Behavioural traces are a joint product of the malware and of the sandbox that records it. The analysis environment writes its own values into the trace, such as user names, file-system paths and machine identifiers, and much malware first checks whether it is running in a virtual machine, a debugger or a sandbox [@afianian2019]. A longitudinal study of 45,375 Windows samples observed between 2010 and 2019 catalogued 92 evasive techniques and found that their prevalence rose slightly over the decade [@galloro2022]. Two installations of the same sandbox can therefore record the same family differently, and a model trained on one installation may learn installation-specific regularities instead of malware behaviour. Kaya et al. showed that behavioural detectors trained on sandbox traces lose much of their accuracy on another sandbox and most of it on endpoint traces, and identified label noise, distribution shift and spurious features such as trace length among the causes [@kaya2025]. Whether the same happens to family attribution, the finding most often reported to investigators, and whether the loss can be traced to identifiable environment tokens has not been measured.

## Motivation and contributions

Four questions therefore remain open. Does family attribution learned in one installation transfer to another installation of the same sandbox? Do tokens that record the environment, or the malware's checks for it, carry the attribution? Can removing those tokens during training restore transfer without losing accuracy within an installation? And can the analysis be fixed and recorded before the results are known, so that its error rates can be audited? Answering them requires two independently operated installations that share families, a fixed classification of trace tokens, interventions rather than post hoc explanations alone, and a verifiable record of the analysis.

Two public datasets now make the question testable without executing malware. MELD-DS-448 releases reports of the CAPE sandbox, version 2 (CAPEv2), and API-call sequences for 26,166 Windows samples from 448 families first seen between 2020 and 2025 [@meld2025]. WinMET releases 31,844 traces from a separately operated CAPEv2 installation on Windows 10, curated by the MALVADA framework [@raducu2025dib; @raducu2025softx]. The two installations share malware families but not machines, configurations or curation.

Here we design and test a framework for collecting behavioural malware evidence and validating deep-learning family attribution across sandbox installations. The work makes four contributions:

1. A layered evidence framework that keeps each trace with its environment record, seals every artefact, model and result in a hash-chained ledger anchored by external time-stamps, and maps each stage to forensic and incident-response standards [@iso27042; @nist80061r3].
2. A frozen lexicon that classes every trace token as behaviour, environment footprint or environment probing, with masking operators and matched random controls that turn reliance on environment artefacts into a measurable intervention.
3. A pre-registered audit of four hypotheses on two independent CAPEv2 installations, with family-balanced metrics, near-duplicate control and cluster-bootstrap inference, applied to a Transformer encoder and three comparison learners.
4. Evidence that family attribution loses 21–53 points of family-balanced macro-averaged F1 score (macro-F1, scale 0–100) between installations, while the identified environment tokens explain almost none of the loss, with the consequences for the error rates attached to forensic findings.

The remainder of the paper is organised as follows. The next section reviews existing methods and the gaps they leave. The Methodology section describes the proposed framework, the evidence-integrity mechanism, the datasets, the learners, the mathematical model and the audit algorithm. The Experimental setup and Results sections report the audit, the Discussion considers what the findings mean for forensic practice, and the final section gives the conclusions, the limitations of the study and future work.

# Review of existing methods

This section reviews learning from behavioural malware evidence, the sources of environment dependence in sandbox traces, evaluation practice and the forensic handling of evidence. Table 1 summarises representative approaches and positions this study among them, Table {{T:lit}} extends the review to 45 studies grouped into thirteen themes, and Table {{T:comp}} compares twenty techniques, tools and datasets on their evidence, evaluation controls and evidence handling.

## Sandbox installations and environment artefacts

Installations of the same sandbox differ in more than their machines. Operators choose the guest operating system and its build, the set of hooked API functions, analysis timeouts, network routing and the processing applied to reports. WinMET, for example, was produced by a CAPEv2 installation with additional hooks, and its release anonymised sensitive values and discarded executions suspected of evasion [@raducu2025dib], whereas MELD-DS-448 releases the reports of a separate installation with its own machines and labels [@meld2025]. Each choice can leave a mark in the tokens a model reads, either as a value set by the environment or as a change in what evasive malware does after probing its surroundings. A model that relies on either can score well within one installation and fail on another, and a validation report written within one installation would not reveal the failure.

## Shortcut learning and explanations

The concern is an instance of shortcut learning, in which decision rules perform well on standard benchmarks but fail under changed conditions [@geirhos2020]. Explanations have exposed such shortcuts in image classifiers [@lapuschkin2019], and deep-learning detectors of COVID-19 in chest radiographs were found to rely on confounding features of their data sources rather than on pathology, and to fail when tested in new hospitals [@degrave2021]. Explanation methods are now widely applied to malware models [@saqib2024; @galli2024; @gulmez2024], and surveys document their spread across security applications [@capuano2022; @rjoub2023; @nadeem2023]. Yet interpreters disagree on the same malware models [@fan2021], their faithfulness has to be tested rather than assumed [@warnecke2020; @samek2021; @nauta2023], and an attribution map cannot show how a model's error would change on evidence from another environment. That question needs an intervention: remove the suspected tokens and measure the change in error within and across environments.

## Evaluation practice

Evaluation practice adds risks of its own. Temporal and spatial bias inflate reported malware-classification results [@pendlebury2019]. Data leakage and spurious correlations recur in security and in machine-learning-based science more generally [@arp2024; @kapoor2023], and malware research faces reproducibility problems that include private datasets and non-deterministic analysis results [@botacin2021]. Concept drift degrades classifiers as malware evolves [@barbero2022; @ceschin2023]. Family labels derived from antivirus names disagree with one another, and datasets with ground-truth families remain small [@joyce2023; @joyce2025]. Each of these problems can produce a confident number that does not transfer. Pre-registration, which fixes hypotheses, analyses and decision rules before outcomes are observed, is an established remedy for analytic flexibility, including for pre-existing data [@nosek2018], and reproducibility checklists are now common in machine learning [@pineau2021]. We found no pre-registered evaluation of a learning-based malware classifier in the literature we reviewed (Table {{T:lit}}).

## Other evidence modalities and forensic soundness

Behavioural evidence is one of several modalities. Static features of portable executable files support benchmark-scale detection, as in EMBER2024 with its 3,238,315 files and temporal test split [@joyce2025], and memory images support detection of obfuscated malware, where temporal evaluation has also proved necessary [@alani2023; @memaldet2024]. Behavioural traces differ in two ways. They record what a sample did, which links them to the tactics and techniques of the MITRE Adversarial Tactics, Techniques and Common Knowledge (ATT&CK) knowledge base used by incident responders [@alsada2025; @sun2024]. They are also the modality most exposed to the analysis environment. Forensic soundness, finally, concerns the evidence and the analysis record together. Hashing, chain of custody and time-stamping make the handling of evidence verifiable [@iso27037], but learning pipelines rarely record which data, model version and decision rules produced a finding, or whether thresholds were fixed before the results were seen.

**Table 1. Representative approaches to learning from malware evidence, compared with this study.**

| Study | Evidence | Learner | Data | Evaluated across time or environments | Evidence handling | Headline result as published |
|:----------|:----------|:------------|:------------|:------------|:----------|:--------------|
| EMBER2024 baseline [@joyce2025] | Static PE features | Gradient-boosted trees | EMBER2024; 3,238,315 files; 2023–2024 | Later weeks; evasive challenge set | SHA-256 and TLSH digests | ROC-AUC 0.9969 on later weeks |
| Kaya et al. [@kaya2025] | Behavioural traces | n-gram, hybrid and attention detectors | Two sandboxes; endpoint traces | Across sandboxes and to endpoints | NR | Above 90% in sandbox evaluations; about 20% at endpoints |
| Nebula [@trizna2024] | Dynamic-analysis reports | Transformer | Three datasets from different platforms | NR | NR | Up to 12% better at low false-positive rates |
| CruParamer [@chen2022] | API calls with parameters | Deep neural network | Two datasets | NR | NR | Better than models on raw API names |
| API-MalDetect [@maniriho2023] | API-call sequences | Deep learning | Several API-call benchmarks | NR | NR | Better than earlier detectors |
| Galli et al. [@galli2024] | API-call sequences | LSTM, GRU; LIME, SHAP, LRP | NR | NR | NR | Explainers compared on behavioural detectors |
| XRan [@gulmez2024] | API, DLL and mutex features | CNN; LIME, SHAP | Ransomware traces | NR | NR | True-positive rate up to 99.4% |
| MeMalDet [@memaldet2024] | Memory dumps | Autoencoders, stacked ensemble | MemMal-D2024 | Temporal split | NR | Accuracy up to 98.82% |
| Beg et al. [@beg2026a] | Evidence graphs | Bayesian meta-learning | Few-shot zero-day episodes | Unseen-family episodes | Forensic signatures | 5-shot accuracy more than 10 points above MAML |
| This study | CAPEv2 API traces with environment records | Transformer, BiLSTM, 1D-CNN, n-gram LightGBM | MELD-DS-448 and WinMET; shared families | Across installations; later months; pre-registered | Hash-chained ledger; RFC 3161 tokens | Macro-F1 47.7 and 62.0 within; 23.3 and 24.1 across |

Values are quoted as published and are not comparable across rows, because tasks (detection, family attribution, explanation), datasets and metrics differ. NR, not reported in the sources read for this study. 1D-CNN, one-dimensional convolutional neural network; API, application programming interface; BiLSTM, bidirectional long short-term memory network; CNN, convolutional neural network; DLL, dynamic-link library; GRU, gated recurrent unit; LIME, local interpretable model-agnostic explanations; LRP, layer-wise relevance propagation; LSTM, long short-term memory network; MAML, model-agnostic meta-learning; PE, portable executable; RFC, Request for Comments; ROC-AUC, area under the receiver-operating-characteristic curve; SHA-256, Secure Hash Algorithm with a 256-bit digest; SHAP, Shapley additive explanations; TLSH, Trend Micro locality-sensitive hash.

## Literature matrix

Table {{T:lit}} summarises 45 studies in the format of the study protocol: title, methods, idea, limitations and the research gap each leaves for the question of this paper. The studies fall into thirteen groups: surveys of Windows malware detection (2), deep learning on behavioural and API-call sequences (3), artificial intelligence in digital forensics (2), evidence-driven detection frameworks from the author’s group (3), evasion and environment-aware malware (4), shortcut learning in other fields (3), explanation methods and their evaluation (10), evaluation pitfalls, drift and rejection (6), family labels and benchmark datasets (5), reproducibility and pre-registration (2), memory forensics (2), mapping of malware to attacker techniques (1) and adversarial machine learning (2). In 21 of the 45 rows the gap concerns the collection environment, the installation or the sandbox, and in 6 it concerns how explanations are checked. No study in the matrix measures attribution error between installations of one sandbox, separates environment tokens from behaviour before training, or seals its analysis record.

Three observations follow from the matrix. First, the behavioural-learning studies treat the sandbox as a fixed instrument: they compare learners on traces from one collection environment [@maniriho2023; @chen2022; @trizna2024], and the surveys that frame the field list transfer between environments as open [@maniriho2024jss; @maniriho2024csur]. Second, explanation research has moved from proposing attributions to evaluating them [@samek2021; @nauta2023; @nadeem2023], and security-specific comparisons show that explanation methods disagree and must be checked [@warnecke2020; @fan2021]; applied studies on API-call and ransomware detectors used such methods to describe model reliance [@galli2024; @gulmez2024], and the matrix records interventional checks and cross-environment validation as the gaps they leave. Third, the evaluation literature supplies the controls that this study applies, including time-aware splits and area under time [@pendlebury2019], pitfalls such as spurious correlations and data snooping [@arp2024], leakage across fields [@kapoor2023] and rejection under drift [@barbero2022], but it addresses shift over time rather than shift between installations. The behavioural datasets that make an installation-level audit possible were released only in 2025 [@raducu2025dib; @raducu2025softx; @meld2025], which may explain why the question has not been asked before.

{{LANDSCAPE_START}}

**Table {{T:lit}}. Literature matrix of learning-based malware analysis, explanation, evaluation practice and evidence handling.**

| S.No. | Title | Methods/Techniques used | Idea used | Limitations | Research gaps | Citation |
|:-----|:--------------------|:----------------------|:--------------------|:----------------|:----------------|:-----|
| 1 | Systematic literature review on Windows malware detection | Systematic review | Taxonomy of static, dynamic and hybrid Windows detection | Review; no new experiments | Transfer of behaviour models between sandboxes | [@maniriho2024jss] |
| 2 | Deep learning models for detecting malware on desktop and mobile platforms | Survey | Organises deep-learning detectors across platforms | Survey; no new experiments | Error rates under changed collection conditions | [@maniriho2024csur] |
| 3 | API-MalDetect | Deep learning on Windows API-call sequences | Automated detection that reports important API calls; data released | Detection (benign versus malicious) only | Environment dependence of API features | [@maniriho2023] |
| 4 | CruParamer | Deep neural network on parameter-augmented API sequences | Call parameters add signal to API names | Detection; two datasets | Parameter values may carry environment-specific values | [@chen2022] |
| 5 | Nebula: self-attention for dynamic malware analysis | Transformer on tokenised dynamic-analysis reports; self-supervised pre-training | One architecture for heterogeneous report formats | Attention-based explanations are descriptive, not interventional | Separating environment artefacts from behaviour | [@trizna2024] |
| 6 | Role of AI and ML in digital forensics and incident response | Analysis of AI and ML across forensic tasks, including chain of custody | Maps AI techniques to forensic tasks and their limits | Review; no new experiments | Validated error rates for learned findings | [@dunsin2024] |
| 7 | Evidence-driven few-shot meta-learning for zero-day detection and forensic attribution | Evidence-graph episodes; Bayesian meta-learning; counterfactual Shapley masks; rule distillation | Learned decisions turned into reusable forensic signatures | Evaluated on constructed few-shot episodes | Validation across collection environments | [@beg2026a] |
| 8 | Multimodal generative cyber forensics network | Temporal alignment, neuro-symbolic hypergraph reasoning, confidence-aware attribution | Unified multimodal forensic intelligence | Accuracy fell under temporally and family-disjoint testing (abstract) | Installation-level shift of behavioural evidence | [@beg2026b] |
| 9 | Evidence drift analytics for long-term malware investigations | Divergence fields, causal lineage graph Transformer, Bayesian survival modelling | Forensic validity of artefacts over long investigations | Several evaluation datasets are not malware-specific | Shift between sandbox installations | [@patidar2026] |
| 10 | Explainable digital forensics AI | Interpretable models for forensic analysis | Interpretability to reduce distrust of forensic AI | Not malware-specific | Error rates under matching conditions | [@solanke2022] |
| 11 | Malware dynamic analysis evasion techniques: a survey | Survey and classification of evasion tactics | Evasion mainly targets sandboxes, chiefly by fingerprinting | Survey; techniques up to 2019 | Effect of evasion on learned models | [@afianian2019] |
| 12 | Evasive behaviours in Windows malware | Evasion-analysis framework; 92 techniques; 45,375 samples, 2010–2019 | Prevalence and evolution of evasive techniques | Measures samples, not trained models | Effect of probing on learned attribution | [@galloro2022] |
| 13 | ML-based behavioral malware detection is far from a solved problem | Sandbox and endpoint traces; n-gram, hybrid and attention detectors; invariance and soft labels | Measures the gap between sandbox and endpoint performance | Detection (benign versus malicious) only | Family attribution; artefact classes; pre-registration | [@kaya2025] |
| 14 | Shortcut learning in deep neural networks | Perspective across fields | Benchmarks reward shortcuts that fail to transfer | Not security-specific | Shortcut tests for malware evidence | [@geirhos2020] |
| 15 | Unmasking Clever Hans predictors | Layer-wise relevance propagation; spectral relevance analysis | Explanations reveal shortcut strategies | Not security-specific | Interventional confirmation in security | [@lapuschkin2019] |
| 16 | AI for radiographic COVID-19 detection selects shortcuts over signal | Explainable AI on chest-radiograph models; testing in new hospitals | Source confounds yield shortcuts that fail at new sites | Medical imaging | Same audit for malware evidence | [@degrave2021] |
| 17 | Explainable AI for malware hunting | Survey | Explainable AI across malware analysis tasks | Survey; no new experiments | Explanations checked by intervention | [@saqib2024] |
| 18 | Explainability in behavioural malware detection systems | LSTM and GRU detectors with LIME, SHAP, LRP and attention | Explainers compared on behavioural models | No ground truth for environment tokens | Reliance on environment tokens | [@galli2024] |
| 19 | XRan | CNN on API, DLL and mutex features; LIME and SHAP | Explainable ransomware detection | Ransomware only | Cross-environment validation | [@gulmez2024] |
| 20 | Explainable AI in cybersecurity: a survey | Survey | Explainable AI across security tasks | Survey; no new experiments | Evaluation standards for explanations | [@capuano2022] |
| 21 | Survey on explainable AI for cybersecurity | Survey | Taxonomy of explainable AI in security | Survey; no new experiments | Evaluation standards for explanations | [@rjoub2023] |
| 22 | Sanity checks for interpreters in Android malware analysis | LIME, Anchor, LORE, SHAP and LEMNA with sanity checks | Interpreters disagree; checks are needed | Android only | Behavioural Windows models | [@fan2021] |
| 23 | Explaining deep neural networks and beyond | Review and comparative evaluation of post hoc explanations | Best practice for explainable AI | General review | Domain-specific validation | [@samek2021] |
| 24 | From anecdotal evidence to quantitative evaluation methods | Systematic review of explanation evaluation | Quantitative properties of explanations | Review; no new experiments | Application to security evidence | [@nauta2023] |
| 25 | Evaluating explanation methods for deep learning in security | Explanation methods compared on security models | Security-specific evaluation criteria | No ground truth for environment artefacts | Interventional evaluation | [@warnecke2020] |
| 26 | SoK: explainable machine learning for computer security | Systematisation of knowledge | Uses and gaps of explainable learning in security | No new experiments | Ground-truth evaluation of explanations | [@nadeem2023] |
| 27 | Pitfalls in machine learning for computer security | Review of 30 papers and experiments | Ten pitfalls, including spurious correlations | Prevalence study | Pitfalls specific to sandbox environments | [@arp2024] |
| 28 | Leakage and the reproducibility crisis in ML-based science | Survey across 17 fields; model info sheets | Leakage inflates reported performance | Not security-specific | Leakage through near-duplicate traces | [@kapoor2023] |
| 29 | Challenges and pitfalls in malware research | Review of 491 papers; 20 pitfalls | A research method that avoids the pitfalls | Review; no new experiments | Environment-aware validation | [@botacin2021] |
| 30 | TESSERACT | Spatial and temporal constraints; area under time | Experimental bias inflates malware classifiers | Android corpus | Bias across collection environments | [@pendlebury2019] |
| 31 | Transcending TRANSCEND | Conformal evaluation with rejection | Classification with rejection under drift | Drift over time, not across environments | Shift between installations | [@barbero2022] |
| 32 | Fast & Furious | Data-stream pipeline with drift detectors | Update features and classifier as malware evolves | Android detection | Behavioural Windows evidence | [@ceschin2023] |
| 33 | MOTIF | Family labels from threat reports | Ground-truth family labels | Smaller than antivirus-labelled corpora | Label agreement across installations | [@joyce2023] |
| 34 | EMBER2024 | 3,238,315 files; temporal split; ClarAVy family labels | Holistic benchmark including families and an evasive set | Static features only | Behavioural installations | [@joyce2025] |
| 35 | The preregistration revolution | Methodology | Pre-registration separates prediction from postdiction, also for existing data | Not domain-specific | Use in malware machine learning | [@nosek2018] |
| 36 | Improving reproducibility in machine learning research | Report on a reproducibility programme | Code policy, challenge and checklist | Machine-learning conferences | Licensed security datasets | [@pineau2021] |
| 37 | XMal | Reduced-feature learning and SHAP on memory features | Lightweight explainable memory detection | Random split (noted by [@memaldet2024]) | Temporal and environment validation | [@alani2023] |
| 38 | MeMalDet | Deep autoencoders and stacked ensemble; temporal evaluation | Memory-based detection under drift | Near-balanced test sets | Collection-environment shift | [@memaldet2024] |
| 39 | MITRE ATT&CK: state of the art and way forward | Survey | ATT&CK as a research instrument | Survey; no new experiments | Probing techniques in learned models | [@alsada2025] |
| 40 | Malware2ATT&CK | Assembly and API embeddings; graph learning; technique mapping | Map malware to ATT&CK techniques | Technique labels are imbalanced | Auditing classifier reliance | [@sun2024] |
| 41 | Adversarial examples in deep learning-enabled cybersecurity systems | Survey of attacks and defences across eight cybersecurity application categories | Taxonomy of adversarial-example attacks and countermeasures | Survey; no new experiments | Adversaries against behavioural attribution | [@macas2024] |
| 42 | Adversarial attacks against Windows PE malware detection | Survey of attacks and defences | Format, executability and maliciousness constraints | Static PE focus | Adversaries against behavioural models | [@ling2023] |
| 43 | A dataset of Windows malware execution traces (WinMET) | CAPEv2 on Windows 10; 31,844 traces; MALVADA processing | Labelled execution traces for reuse | No benchmark model | Cross-installation use | [@raducu2025dib] |
| 44 | MALVADA | Framework that filters, structures and labels CAPEv2 traces | Reproducible dataset generation | One installation | Harmonisation across installations | [@raducu2025softx] |
| 45 | MELD-DS-448 | CAPE reports and API sequences; 26,166 samples; 448 families | Long-tailed family dataset, 2020–2025 | No published benchmark | Cross-installation use | [@meld2025] |

Columns follow the study protocol. Entries summarise the published abstracts and, where noted in the Reference Verification Log, full texts; they describe scope relative to the question of this study, not the quality of the work. Research gaps are stated with respect to the transfer of learned findings between installations.

{{LANDSCAPE_END}}

## Comparison of techniques, tools and datasets

Table {{T:comp}} compares twenty techniques, tools and datasets, including this study, on evidence modality, learner family, dataset, evidence handling, bias control, reported metric, false-positive rate, generalisation evidence, artefact availability and the gap that remains. Of the 19 published entries, 15 are models or evaluation frameworks and 4 are datasets or measurement studies. Only three of the 15 used a bias-controlled evaluation: time-ordered evaluation of Android malware [@pendlebury2019; @ceschin2023] and evaluation across sandboxes and endpoints [@kaya2025]. Another three used temporal splits with balanced test sets or a random split, and nine did not report the split. Evaluation across collection environments was reported only by Kaya et al., and for detection rather than family attribution.

The other columns show the same pattern. A false-positive rate or a low-FPR operating point was reported for three entries [@joyce2025; @kaya2025; @trizna2024], although an error rate at a stated operating point is what a forensic report needs. Evidence handling was reported for five entries: the hashes, curation steps or report-based labels published with four benchmarks and datasets, EMBER2024 [@joyce2025], MOTIF [@joyce2023], WinMET [@raducu2025dib] and MELD-DS-448 [@meld2025], and forensic signatures produced by one model [@beg2026a]; none sealed models and results in a verifiable record. Generalisation was most often tested over time (six entries), and nine entries released data, code, models or a leaderboard, while 10 did not report an artefact. Because the reported metrics range from ROC-AUC and accuracy to explanation stability, the table compares what each study controlled and reported rather than which method scored highest. Against these entries, this study is the only one that combines behavioural evidence with an environment record, a pre-registered evaluation across installations with near-duplicate control, and a hash-chained, time-stamped record of data, models and results.

{{LANDSCAPE_START}}

**Table {{T:comp}}. Comparison of twenty techniques, tools and datasets for learning from malware evidence.**

| Technique | Evidence modality | ML/DL family | Dataset (year) | Forensic/evidence handling | Bias-controlled evaluation | Reported metric | FPR | Generalisation evidence | Artefact available | Research gap |
|:-----------|:--------|:----------|:----------|:----------|:----------|:---------|:------|:---------|:--------|:----------|
| EMBER2024 LightGBM [@joyce2025] | Static PE | Gradient-boosted trees | EMBER2024 (2023–2024) | SHA-256, TLSH; within-week de-duplication | N (temporal split; balanced test) | ROC-AUC 0.9969 | TPR 94.48% at 1% FPR | Later weeks; evasive set | Yes | No environment-dependence analysis |
| Kaya et al. [@kaya2025] | Behavioural | n-gram, hybrid, attention | Two sandboxes; endpoints | NR | Y (across environments) | TPR at 1% FPR | 1% | Across sandboxes and to endpoints | Leaderboard | Detection only; artefact classes not separated |
| Nebula [@trizna2024] | Dynamic reports | Transformer | Three datasets | NR | NR | Detection and classification | Low-FPR regime | Several platforms | Code, models | Interventional test of artefacts |
| CruParamer [@chen2022] | API with parameters | Deep neural network | Two datasets | NR | NR | Detection | NR | NR | NR | Parameter values and environment |
| API-MalDetect [@maniriho2023] | API sequences | Deep learning | Several API benchmarks | NR | NR | Accuracy, F1, ROC-AUC | NR | Several datasets | Data | Environment dependence |
| Galli et al. [@galli2024] | API sequences | LSTM, GRU; post hoc XAI | NR | NR | NR | Explanation comparison | NR | NR | NR | Explanations not interventional |
| XRan [@gulmez2024] | Dynamic (API, DLL, mutex) | CNN; LIME, SHAP | Ransomware traces | NR | NR | TPR up to 99.4% | NR | NR | NR | Cross-environment validation |
| MeMalDet [@memaldet2024] | Memory | Autoencoder, stacked ensemble | MemMal-D2024 | NR | N (temporal; near-balanced) | Accuracy up to 98.82% | NR | Later data | Code | Base rate; custody |
| XMal [@alani2023] | Memory | Reduced-feature learning; SHAP | Memory features | NR | N (random split) | Accuracy above 99% | NR | NR | NR | Temporal validation |
| Malware2ATT&CK [@sun2024] | Code and API | Graph learning | NR | NR | NR | Technique F1 | NR | NR | NR | Use for auditing classifiers |
| MOTIF [@joyce2023] | Static PE (dataset) | None (dataset) | MOTIF | Report-based labels | N/A | N/A | N/A | N/A | Yes | Size |
| WinMET and MALVADA [@raducu2025dib; @raducu2025softx] | Behavioural (dataset) | None (dataset) | WinMET (2025 release) | Curation, anonymisation, labels | N/A | N/A | N/A | N/A | Yes | No benchmark model |
| MELD-DS-448 [@meld2025] | Behavioural (dataset) | None (dataset) | MELD-DS-448 (2020–2025) | Manifest with hashes | N/A | N/A | N/A | N/A | Yes | No benchmark model |
| Evidence meta-learning [@beg2026a] | Evidence graphs | Bayesian meta-learning | Few-shot episodes | Forensic signatures | NR | 5-shot accuracy | NR | Unseen-family episodes | NR | Collection-environment shift |
| Fast & Furious [@ceschin2023] | Android static | Data-stream learners | DREBIN, AndroZoo (2009–2018) | NR | Y (time-ordered) | F1 | NR | Across time | Code, data | Windows behaviour |
| TRANSCEND revisited [@barbero2022] | Malware features | Conformal rejection | NR | NR | NR | Rejection-aware performance | NR | Across time | NR | Environment shift |
| TESSERACT [@pendlebury2019] | Android features | Evaluation framework | Android apps | NR | Y (by design) | Area under time | N/A | Across time | NR | Environment shift |
| Evasion study [@galloro2022] | Binaries (evasion analysis) | None (measurement) | 45,375 samples (2010–2019) | NR | N/A | Prevalence | N/A | Across years | NR | Effect on learned models |
| Interpreter sanity checks [@fan2021] | Android | Five interpreters | Android malware | NR | NR | Explanation stability | N/A | NR | NR | Behavioural Windows models |
| This study | Behavioural with environment record | Transformer, BiLSTM, 1D-CNN, n-gram LightGBM | MELD-DS-448, WinMET (2020–2025) | Hash-chained ledger; RFC 3161 tokens | Y (later months; near-duplicate control; pre-registered) | Family-balanced macro-F1 | N/A (attribution) | Across installations | Code, derived data | Replay identical binaries |

Bias-controlled evaluation (Y/N): a time-aware split with both classes from the same windows plus a realistic base rate or a cross-split duplicate control, or an evaluation across collection environments. NR, not reported in the sources read; N/A, not applicable. Values are as published and are not comparable across rows. ML/DL, machine learning/deep learning; FPR, false-positive rate; TPR, true-positive rate; PE, portable executable; GRU, gated recurrent unit; LIME, local interpretable model-agnostic explanations; SHAP, Shapley additive explanations; DLL, dynamic-link library; TLSH, Trend Micro locality-sensitive hash.

{{LANDSCAPE_END}}

## Research gaps

The approaches in Tables 1, {{T:lit}} and {{T:comp}} leave five gaps, which this study addresses:

1. Transfer of family attribution between installations has not been measured. Behavioural detectors lose accuracy on another sandbox and on endpoint traces [@kaya2025], but whether family attribution, the finding most often reported to investigators, loses accuracy between installations of the same sandbox, and whether the loss can be traced to identifiable environment tokens, is unknown.
2. Environment artefacts are not separated from behaviour. Installation choices leave marks in the tokens a model reads [@raducu2025dib; @meld2025], and a model that relies on them can score well within one installation and fail on another without a within-installation validation revealing it [@geirhos2020].
3. Explanations alone cannot settle the question. Explanation methods are widely applied to malware models [@saqib2024; @galli2024; @gulmez2024], but interpreters disagree [@fan2021], their faithfulness must be tested [@warnecke2020; @nauta2023], and an attribution map cannot show how the error would change on evidence from another environment; that requires an intervention.
4. Evaluations are rarely fixed in advance. Temporal and spatial bias, leakage and drift can produce confident numbers that do not transfer [@pendlebury2019; @arp2024; @ceschin2023], and no pre-registered evaluation of a learning-based malware classifier was found in the literature reviewed (Table {{T:lit}}).
5. The analysis record is rarely forensically sound. Hashing, chain of custody and time-stamping make the handling of evidence verifiable [@iso27037], but learning pipelines rarely record which data, model version and decision rules produced a finding, and most approaches in Table 1 report no evidence handling.

## Positioning of this study

The framework proposed here addresses the five gaps together. It measures family attribution within and across two independently operated installations of the same sandbox (gap 1). A frozen lexicon classes every trace token as behaviour, footprint or probing before any model is trained (gap 2). Masking at test time and during training tests the token classes directly, with explanations as a secondary check (gap 3). Hypotheses, margins, splits and near-duplicate rules were registered and time-stamped before any test trace was scored (gap 4), and every artefact, model and result is sealed in a hash-chained ledger anchored by trusted time-stamps (gap 5).
