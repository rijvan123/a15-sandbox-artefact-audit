"""Supplementary Information for the A15 manuscript. Reference keys use the main reference numbering."""
from __future__ import annotations

import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITLE = ("Deep learning attribution of malware families from sandbox behaviour traces transfers poorly between "
         "installations of the same sandbox")

# (key, short title, methods/techniques, idea used, limitations, research gap)
S1 = [
 ("maniriho2024jss", "Systematic literature review on Windows malware detection", "Systematic review", "Taxonomy of static, dynamic and hybrid Windows detection", "Review; no new experiments", "Transfer of behaviour models between sandboxes"),
 ("maniriho2024csur", "Deep learning models for detecting malware on desktop and mobile platforms", "Survey", "Organises deep-learning detectors across platforms", "Survey; no new experiments", "Error rates under changed collection conditions"),
 ("maniriho2023", "API-MalDetect", "Deep learning on Windows API-call sequences", "Automated detection that reports important API calls; data released", "Detection (benign versus malicious) only", "Environment dependence of API features"),
 ("chen2022", "CruParamer", "Deep neural network on parameter-augmented API sequences", "Call parameters add signal to API names", "Detection; two datasets", "Parameter values may carry environment-specific values"),
 ("trizna2024", "Nebula: self-attention for dynamic malware analysis", "Transformer on tokenised dynamic-analysis reports; self-supervised pre-training", "One architecture for heterogeneous report formats", "Attention-based explanations are descriptive, not interventional", "Separating environment artefacts from behaviour"),
 ("dunsin2024", "Role of AI and ML in digital forensics and incident response", "Analysis of AI and ML across forensic tasks, including chain of custody", "Maps AI techniques to forensic tasks and their limits", "Review; no new experiments", "Validated error rates for learned findings"),
 ("beg2026a", "Evidence-driven few-shot meta-learning for zero-day detection and forensic attribution", "Evidence-graph episodes; Bayesian meta-learning; counterfactual Shapley masks; rule distillation", "Learned decisions turned into reusable forensic signatures", "Evaluated on constructed few-shot episodes", "Validation across collection environments"),
 ("beg2026b", "Multimodal generative cyber forensics network", "Temporal alignment, neuro-symbolic hypergraph reasoning, confidence-aware attribution", "Unified multimodal forensic intelligence", "Accuracy fell under temporally and family-disjoint testing (abstract)", "Installation-level shift of behavioural evidence"),
 ("patidar2026", "Evidence drift analytics for long-term malware investigations", "Divergence fields, causal lineage graph Transformer, Bayesian survival modelling", "Forensic validity of artefacts over long investigations", "Several evaluation datasets are not malware-specific", "Shift between sandbox installations"),
 ("solanke2022", "Explainable digital forensics AI", "Interpretable models for forensic analysis", "Interpretability to reduce distrust of forensic AI", "Not malware-specific", "Error rates under matching conditions"),
 ("afianian2019", "Malware dynamic analysis evasion techniques: a survey", "Survey and classification of evasion tactics", "Evasion mainly targets sandboxes, chiefly by fingerprinting", "Survey; techniques up to 2019", "Effect of evasion on learned models"),
 ("galloro2022", "Evasive behaviours in Windows malware", "Evasion-analysis framework; 92 techniques; 45,375 samples, 2010–2019", "Prevalence and evolution of evasive techniques", "Measures samples, not trained models", "Effect of probing on learned attribution"),
 ("kaya2025", "ML-based behavioral malware detection is far from a solved problem", "Sandbox and endpoint traces; n-gram, hybrid and attention detectors; invariance and soft labels", "Measures the gap between sandbox and endpoint performance", "Detection (benign versus malicious) only", "Family attribution; artefact classes; pre-registration"),
 ("geirhos2020", "Shortcut learning in deep neural networks", "Perspective across fields", "Benchmarks reward shortcuts that fail to transfer", "Not security-specific", "Shortcut tests for malware evidence"),
 ("lapuschkin2019", "Unmasking Clever Hans predictors", "Layer-wise relevance propagation; spectral relevance analysis", "Explanations reveal shortcut strategies", "Not security-specific", "Interventional confirmation in security"),
 ("degrave2021", "AI for radiographic COVID-19 detection selects shortcuts over signal", "Explainable AI on chest-radiograph models; testing in new hospitals", "Source confounds yield shortcuts that fail at new sites", "Medical imaging", "Same audit for malware evidence"),
 ("saqib2024", "Explainable AI for malware hunting", "Survey", "Explainable AI across malware analysis tasks", "Survey; no new experiments", "Explanations checked by intervention"),
 ("galli2024", "Explainability in behavioural malware detection systems", "LSTM and GRU detectors with LIME, SHAP, LRP and attention", "Explainers compared on behavioural models", "No ground truth for environment tokens", "Reliance on environment tokens"),
 ("gulmez2024", "XRan", "CNN on API, DLL and mutex features; LIME and SHAP", "Explainable ransomware detection", "Ransomware only", "Cross-environment validation"),
 ("capuano2022", "Explainable AI in cybersecurity: a survey", "Survey", "Explainable AI across security tasks", "Survey; no new experiments", "Evaluation standards for explanations"),
 ("rjoub2023", "Survey on explainable AI for cybersecurity", "Survey", "Taxonomy of explainable AI in security", "Survey; no new experiments", "Evaluation standards for explanations"),
 ("fan2021", "Sanity checks for interpreters in Android malware analysis", "LIME, Anchor, LORE, SHAP and LEMNA with sanity checks", "Interpreters disagree; checks are needed", "Android only", "Behavioural Windows models"),
 ("samek2021", "Explaining deep neural networks and beyond", "Review and comparative evaluation of post hoc explanations", "Best practice for explainable AI", "General review", "Domain-specific validation"),
 ("nauta2023", "From anecdotal evidence to quantitative evaluation methods", "Systematic review of explanation evaluation", "Quantitative properties of explanations", "Review; no new experiments", "Application to security evidence"),
 ("warnecke2020", "Evaluating explanation methods for deep learning in security", "Explanation methods compared on security models", "Security-specific evaluation criteria", "No ground truth for environment artefacts", "Interventional evaluation"),
 ("nadeem2023", "SoK: explainable machine learning for computer security", "Systematisation of knowledge", "Uses and gaps of explainable learning in security", "No new experiments", "Ground-truth evaluation of explanations"),
 ("arp2024", "Pitfalls in machine learning for computer security", "Review of 30 papers and experiments", "Ten pitfalls, including spurious correlations", "Prevalence study", "Pitfalls specific to sandbox environments"),
 ("kapoor2023", "Leakage and the reproducibility crisis in ML-based science", "Survey across 17 fields; model info sheets", "Leakage inflates reported performance", "Not security-specific", "Leakage through near-duplicate traces"),
 ("botacin2021", "Challenges and pitfalls in malware research", "Review of 491 papers; 20 pitfalls", "A research method that avoids the pitfalls", "Review; no new experiments", "Environment-aware validation"),
 ("pendlebury2019", "TESSERACT", "Spatial and temporal constraints; area under time", "Experimental bias inflates malware classifiers", "Android corpus", "Bias across collection environments"),
 ("barbero2022", "Transcending TRANSCEND", "Conformal evaluation with rejection", "Classification with rejection under drift", "Drift over time, not across environments", "Shift between installations"),
 ("ceschin2023", "Fast & Furious", "Data-stream pipeline with drift detectors", "Update features and classifier as malware evolves", "Android detection", "Behavioural Windows evidence"),
 ("joyce2023", "MOTIF", "Family labels from threat reports", "Ground-truth family labels", "Smaller than antivirus-labelled corpora", "Label agreement across installations"),
 ("joyce2025", "EMBER2024", "3,238,315 files; temporal split; ClarAVy family labels", "Holistic benchmark including families and an evasive set", "Static features only", "Behavioural installations"),
 ("nosek2018", "The preregistration revolution", "Methodology", "Pre-registration separates prediction from postdiction, also for existing data", "Not domain-specific", "Use in malware machine learning"),
 ("pineau2021", "Improving reproducibility in machine learning research", "Report on a reproducibility programme", "Code policy, challenge and checklist", "Machine-learning conferences", "Licensed security datasets"),
 ("alani2023", "XMal", "Reduced-feature learning and SHAP on memory features", "Lightweight explainable memory detection", "Random split (noted by [@memaldet2024])", "Temporal and environment validation"),
 ("memaldet2024", "MeMalDet", "Deep autoencoders and stacked ensemble; temporal evaluation", "Memory-based detection under drift", "Near-balanced test sets", "Collection-environment shift"),
 ("alsada2025", "MITRE ATT&CK: state of the art and way forward", "Survey", "ATT&CK as a research instrument", "Survey; no new experiments", "Probing techniques in learned models"),
 ("sun2024", "Malware2ATT&CK", "Assembly and API embeddings; graph learning; technique mapping", "Map malware to ATT&CK techniques", "Technique labels are imbalanced", "Auditing classifier reliance"),
 ("macas2024", "Adversarial examples in deep learning-enabled cybersecurity systems", "Survey of attacks and defences across eight cybersecurity application categories", "Taxonomy of adversarial-example attacks and countermeasures", "Survey; no new experiments", "Adversaries against behavioural attribution"),
 ("ling2023", "Adversarial attacks against Windows PE malware detection", "Survey of attacks and defences", "Format, executability and maliciousness constraints", "Static PE focus", "Adversaries against behavioural models"),
 ("raducu2025dib", "A dataset of Windows malware execution traces (WinMET)", "CAPEv2 on Windows 10; 31,844 traces; MALVADA processing", "Labelled execution traces for reuse", "No benchmark model", "Cross-installation use"),
 ("raducu2025softx", "MALVADA", "Framework that filters, structures and labels CAPEv2 traces", "Reproducible dataset generation", "One installation", "Harmonisation across installations"),
 ("meld2025", "MELD-DS-448", "CAPE reports and API sequences; 26,166 samples; 448 families", "Long-tailed family dataset, 2020–2025", "No published benchmark", "Cross-installation use"),
]

# technique, modality, ML family, dataset (year), evidence handling, bias-controlled, metric, FPR, generalisation, artefact, gap
S2 = [
 ("EMBER2024 LightGBM [@joyce2025]", "Static PE", "Gradient-boosted trees", "EMBER2024 (2023–2024)", "SHA-256, TLSH; within-week de-duplication", "N (temporal split; balanced test)", "ROC-AUC 0.9969", "TPR 94.48% at 1% FPR", "Later weeks; evasive set", "Yes", "No environment-dependence analysis"),
 ("Kaya et al. [@kaya2025]", "Behavioural", "n-gram, hybrid, attention", "Two sandboxes; endpoints", "NR", "Y (across environments)", "TPR at 1% FPR", "1%", "Across sandboxes and to endpoints", "Leaderboard", "Detection only; artefact classes not separated"),
 ("Nebula [@trizna2024]", "Dynamic reports", "Transformer", "Three datasets", "NR", "NR", "Detection and classification", "Low-FPR regime", "Several platforms", "Code, models", "Interventional test of artefacts"),
 ("CruParamer [@chen2022]", "API with parameters", "Deep neural network", "Two datasets", "NR", "NR", "Detection", "NR", "NR", "NR", "Parameter values and environment"),
 ("API-MalDetect [@maniriho2023]", "API sequences", "Deep learning", "Several API benchmarks", "NR", "NR", "Accuracy, F1, ROC-AUC", "NR", "Several datasets", "Data", "Environment dependence"),
 ("Galli et al. [@galli2024]", "API sequences", "LSTM, GRU; post hoc XAI", "NR", "NR", "NR", "Explanation comparison", "NR", "NR", "NR", "Explanations not interventional"),
 ("XRan [@gulmez2024]", "Dynamic (API, DLL, mutex)", "CNN; LIME, SHAP", "Ransomware traces", "NR", "NR", "TPR up to 99.4%", "NR", "NR", "NR", "Cross-environment validation"),
 ("MeMalDet [@memaldet2024]", "Memory", "Autoencoder, stacked ensemble", "MemMal-D2024", "NR", "N (temporal; near-balanced)", "Accuracy up to 98.82%", "NR", "Later data", "Code", "Base rate; custody"),
 ("XMal [@alani2023]", "Memory", "Reduced-feature learning; SHAP", "Memory features", "NR", "N (random split)", "Accuracy above 99%", "NR", "NR", "NR", "Temporal validation"),
 ("Malware2ATT&CK [@sun2024]", "Code and API", "Graph learning", "NR", "NR", "NR", "Technique F1", "NR", "NR", "NR", "Use for auditing classifiers"),
 ("MOTIF [@joyce2023]", "Static PE (dataset)", "None (dataset)", "MOTIF", "Report-based labels", "N/A", "N/A", "N/A", "N/A", "Yes", "Size"),
 ("WinMET and MALVADA [@raducu2025dib; @raducu2025softx]", "Behavioural (dataset)", "None (dataset)", "WinMET (2025 release)", "Curation, anonymisation, labels", "N/A", "N/A", "N/A", "N/A", "Yes", "No benchmark model"),
 ("MELD-DS-448 [@meld2025]", "Behavioural (dataset)", "None (dataset)", "MELD-DS-448 (2020–2025)", "Manifest with hashes", "N/A", "N/A", "N/A", "N/A", "Yes", "No benchmark model"),
 ("Evidence meta-learning [@beg2026a]", "Evidence graphs", "Bayesian meta-learning", "Few-shot episodes", "Forensic signatures", "NR", "5-shot accuracy", "NR", "Unseen-family episodes", "NR", "Collection-environment shift"),
 ("Fast & Furious [@ceschin2023]", "Android static", "Data-stream learners", "DREBIN, AndroZoo (2009–2018)", "NR", "Y (time-ordered)", "F1", "NR", "Across time", "Code, data", "Windows behaviour"),
 ("TRANSCEND revisited [@barbero2022]", "Malware features", "Conformal rejection", "NR", "NR", "NR", "Rejection-aware performance", "NR", "Across time", "NR", "Environment shift"),
 ("TESSERACT [@pendlebury2019]", "Android features", "Evaluation framework", "Android apps", "NR", "Y (by design)", "Area under time", "N/A", "Across time", "NR", "Environment shift"),
 ("Evasion study [@galloro2022]", "Binaries (evasion analysis)", "None (measurement)", "45,375 samples (2010–2019)", "NR", "N/A", "Prevalence", "N/A", "Across years", "NR", "Effect on learned models"),
 ("Interpreter sanity checks [@fan2021]", "Android", "Five interpreters", "Android malware", "NR", "NR", "Explanation stability", "N/A", "NR", "NR", "Behavioural Windows models"),
 ("This study", "Behavioural with environment record", "Transformer, BiLSTM, 1D-CNN, n-gram LightGBM", "MELD-DS-448, WinMET (2020–2025)", "Hash-chained ledger; RFC 3161 tokens", "Y (later months; near-duplicate control; pre-registered)", "Family-balanced macro-F1", "N/A (attribution)", "Across installations", "Code, derived data", "Replay identical binaries"),
]

S5 = [
 ("Transformer encoder", "2–4 layers; 4 heads; width 128; feed-forward 256 or 512; dropout; learning rate; weight decay; batch 32; γ ∈ {0, 0.5, 1}; artefact-dropout rate p ∈ {0.25, 0.5, 0.75, 1}", "20 random configurations × 5 seeds; confirmatory"),
 ("BiLSTM with attention", "1–2 layers; width 128; same optimiser grid", "Unmasked and F-masked training only (CPU plan)"),
 ("1D-CNN", "Kernel widths 3, 5, 7; 64 or 128 channels; dropout; learning rate; weight decay", "20 configurations × 5 seeds"),
 ("n-gram LightGBM", "TF-IDF of token 1–3-grams; number of leaves; learning rate; number of trees; minimum child samples; γ", "Full design"),
 ("Probes", "Length-only and artefact-only: gradient-boosted trees; environment-ID: logistic regression on z-scored frozen embeddings", "Same splits as the learners"),
 ("Sequences", "Head-plus-tail; cap 256 tokens; run-length bucket min(⌊log~2~ n~run~⌋, 7); types in fewer than 5 training traces mapped to unknown", "Frozen before scoring"),
 ("Training", "At most 20 epochs; early stopping after 3 epochs without validation improvement; seeds 11, 23, 37, 41, 53", "Search seed fixed"),
 ("Explanations", "Integrated Gradients; logit target; mask-token baseline; Gauss–Legendre m = 50 (100, 200 if needed); 200 traces per cell", "Checks P1 and P2"),
 ("Statistics", "Cluster bootstrap, 9,999 replicates; one-sided α 0.025 per claim; Holm over four hypotheses; Δ = 3 points; ρ bound 0.10", "Frozen in settings.json"),
 ("Hardware and software", "Windows Subsystem for Linux, Ubuntu 24.04; 12 logical CPUs; no GPU; Python 3.12.16; PyTorch 2.5.1 (CPU); NumPy 2.1.3; SciPy 1.14.1; scikit-learn 1.5.2; pandas 2.2.3; LightGBM 4.6.0; ijson 3.3.0; py7zr 0.22.0; Matplotlib 3.9.2", "Projected 58.3 CPU hours (dry run)"),
]

S8 = [('Base model', 'MELD-DS-448', 'MELD-DS-448', '47.7 (44.3–50.3)'),
 ('Base model', 'WinMET', 'WinMET', '62.0 (46.6–65.9)'),
 ('Base model', 'MELD-DS-448', 'WinMET', '23.3 (21.5–25.1)'),
 ('Base model', 'WinMET', 'MELD-DS-448', '24.1 (22.5–25.3)'),
 ('Footprint tokens masked in training', 'MELD-DS-448', 'WinMET', '26.3 (24.2–28.0)'),
 ('Footprint tokens masked in training', 'WinMET', 'MELD-DS-448', '23.9 (22.2–25.6)'),
 ('Footprint and probing tokens masked at test', 'MELD-DS-448', 'MELD-DS-448', '43.9'),
 ('Matched random mask at test', 'MELD-DS-448', 'MELD-DS-448', '43.9'),
 ('Footprint and probing tokens masked at test', 'WinMET', 'WinMET', '57.3'),
 ('Matched random mask at test', 'WinMET', 'WinMET', '56.8'),
 ('Artefact-only probe (behaviour tokens masked)', 'MELD-DS-448', 'MELD-DS-448', '40.2 (36.5–42.9)'),
 ('Length-only probe', 'MELD-DS-448', 'MELD-DS-448', '32.0 (28.8–34.8)'),
 ('Artefact-only probe (behaviour tokens masked)', 'WinMET', 'WinMET', '50.7 (43.6–54.1)'),
 ('Length-only probe', 'WinMET', 'WinMET', '48.4 (39.8–55.4)')]

S6 = [
 ("ψ~1~, MELD-DS-448", "8.2 (4.4 to 11.5)", "†", "7.0 (3.5 to 10.3)", "10.5 (6.5 to 14.2)"),
 ("ψ~1~, WinMET", "2.4 (−6.9 to 9.8)", "†", "7.3 (−5.6 to 14.6)", "13.6 (−4.2 to 23.0)"),
 ("ψ~2~, MELD-DS-448", "−0.03 (−0.96 to 0.95)", "†", "†", "†"),
 ("ψ~2~, WinMET", "−0.47 (−2.17 to 1.35)", "†", "†", "18.4 (−2.1 to 22.3)"),
 ("ψ~3~, MELD-DS-448 to WinMET", "24.4 (20.5 to 27.6)", "†", "†", "†"),
 ("ψ~3~, WinMET to MELD-DS-448", "38.0 (22.6 to 42.2)", "†", "†", "†"),
 ("ρ, pooled", "0.046 (0.014 to 0.084)", "†", "†", "†"),
 ("ψ~4b~, MELD-DS-448", "−0.2 (−1.3 to 0.7)", "†", "†", "†"),
 ("ψ~4b~, WinMET", "−0.9 (−2.9 to 1.4)", "†", "†", "−6.5 (−13.2 to −3.7)"),
]

S7 = [
 ("Registered antimode (0.535, 0.532, 0.528)", "167, 127, 157", "32, 34, 31", "Inconclusive", "Inconclusive", "Inconclusive: 14.5 (4.0 to 20.5) and 16.6 (7.7 to 33.0)", "Not tested"),
 ("0.7", "418, 343, 313", "204, 208, 205", "Inconclusive", "Inconclusive", "Supported: 22.3 and 33.0", "Rejected"),
 ("0.8 (primary)", "705, 589, 508", "322, 329, 324", "Inconclusive", "Rejected", "Supported: 24.4 and 38.0", "Rejected"),
 ("0.9", "955, 816, 633", "838, 855, 845", "Inconclusive", "Rejected", "Supported: 22.2 and 42.2", "Rejected"),
]


def table(header, rows, widths=None):
    widths = widths or [10] * len(header)
    head = "| " + " | ".join(header) + " |\n|" + "|".join(":" + "-" * w for w in widths) + "|"
    return head + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


def notation(eqmap):
    t = (HERE / "assets" / "notation_S8.md").read_text(encoding="utf-8")
    rows = [l for l in t.splitlines() if l.startswith("|")]

    def remap(cell):
        out = []
        for tok in [x.strip() for x in cell.split(",")]:
            if tok == "10":
                out.append("S10")
            elif tok.isdigit() and tok in eqmap:
                out.append(str(eqmap[tok]))
            else:
                out.append(tok)
        return ", ".join(out)

    fixed = []
    for i, l in enumerate(rows):
        cells = [c.strip() for c in l.strip("|").split("|")]
        if i >= 2 and len(cells) == 3:
            cells[2] = remap(cells[2])
        fixed.append("| " + " | ".join(cells) + " |")
    return "\n".join(fixed)


def provenance():
    log = Path("/mnt/project/data_provenance_log.csv")
    rows = []
    if log.exists():
        with open(log, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["status"] == "ok" and r["file"] != "(git tree)":
                    rows.append((r["source"], r["file"], f"{int(r['size_bytes']):,}" if r["size_bytes"] else "",
                                 (r["published_algo"] or "SHA-256") + (" match" if r["digest_match"] == "True" else ""),
                                 r["licence"], r["access_utc"][:10]))
    rows.append(("MALVADA", "git tree, commit 24974027", "", "Commit digest", "GPL-3.0", "2026-11-02"))
    rows.append(("EMBER2024 (labels only)", "6 Windows feature archives, revision 3d23efef", "about 23.1 GB in total",
                 "Verified by the ember-labels stage", "Apache-2.0", "2026-11-04"))
    return table(["Source", "File", "Size (bytes)", "Integrity check", "Licence", "Accessed"], rows, [8, 22, 8, 10, 8, 6])


def supp_equations(eqmap, eqs):
    t = (HERE / "assets" / "suppeq_S8.md").read_text(encoding="utf-8")
    t = re.sub(r"^## 8 Supplementary Equations S1–S9\s*", "", t)
    t = re.sub(r"equation \((\d+)\)", lambda m: f"equation ({eqmap.get(m.group(1), 'S10' if m.group(1) == '10' else m.group(1))})", t)
    t = re.sub(r"equations \((\d+)\)", lambda m: f"equations ({eqmap.get(m.group(1), m.group(1))})", t)
    t += ("\n\n**Weighted Jaccard overlap between installations (mechanism check P3).** For token class C with "
          "relative frequencies Q^(I)^~C~ in installation I, the weighted Jaccard overlap equals (1 − TV)/(1 + TV), "
          "with TV the total-variation distance:\n\n$$\n" + eqs["10"] + "\\qquad(\\mathrm{S}10)\n$$\n")
    return t


def figS1():
    png = HERE / "figs" / "FigS1.png"
    return f"![]({png.as_posix()}){{width=20cm}}\n\n" if png.exists() else ""


def build(order, eqmap, eqs):
    def num(k):
        return f"[{order[k]}]"

    s1 = [(i + 1, t, m, idea, lim, gap, num(k)) for i, (k, t, m, idea, lim, gap) in enumerate(S1)]
    pseudo = (HERE / "assets" / "pseudocode_S7.md").read_text(encoding="utf-8")
    md = f"""::: {{custom-style="Title"}}
Supplementary Information
:::

**{TITLE}**

[Author 1], [Author 2], [Author 3] and [Author 4]

Reference numbers in square brackets refer to the reference list of the main article. This file contains Supplementary Tables S1–S6, Supplementary Figure S1, Supplementary Equations S1–S10 and Supplementary Notes.

# Supplementary Table S1. Notation

Equation numbers refer to the main text; S-numbers refer to the Supplementary Equations.

{notation(eqmap)}

# Supplementary Table S2. Data Provenance Log (files used)

Files downloaded and checked on the author's computer; files skipped by design (disassembly, Markdown summaries and split copies) and the attempts blocked by the network allowlist of the assistant's environment are listed in the full log released with the code.

{provenance()}

# Supplementary Table S3. Search spaces, settings and computing environment

The configuration selected for each learner, cell and seed is recorded in the results file of the evaluation.

{table(["Component", "Settings", "Notes"], S5, [10, 50, 16])}

# Supplementary Table S4. Exploratory learners

Estimates in family-balanced macro-F1 points (ρ as a share) with 95% cluster-bootstrap intervals, at the primary near-duplicate threshold. The exploratory learners were not adjusted for multiplicity. —, not estimated: under the reduced compute plan the BiLSTM was trained unmasked and footprint-masked only, without probes.

{table(["Component", "Transformer (confirmatory)", "BiLSTM", "1D-CNN", "n-gram LightGBM"], S6, [14, 12, 8, 10, 10])}

Ranges reported in the main text: transfer gaps 21.4–27.3 points (MELD-DS-448 to WinMET) and 38.0–52.8 points (WinMET to MELD-DS-448) across learners; recovered shares −0.02 to 0.05 for the exploratory learners; masking footprint tokens alone lowered the n-gram LightGBM's WinMET macro-F1 from 70.8 to 40.1 points.

# Supplementary Table S5. Sensitivity to the near-duplicate threshold

Test traces kept per origin (origins 1–3) and verdicts for H1–H4 at each threshold, from the same trained models. H3 entries give the two transfer gaps in points (95% intervals where reported).

{table(["Threshold τ~J~", "MELD-DS-448 traces kept", "WinMET traces kept", "H1", "H2", "H3", "H4"], S7, [12, 10, 10, 8, 8, 22, 8])}

# Supplementary Table S6. Macro-F1 of the Transformer encoder and the probes under each evaluation condition

Macro-F1 is family-balanced, in points (scale 0–100), averaged over five training seeds and the three rolling origins at the primary near-duplicate threshold of 0.8. Intervals are 95% cluster-bootstrap percentile intervals from 9,999 replicates; for the masking conditions, the interval of the contrast ψ~2~ is given in Table {{T:hyp}}. The length-only probe sees trace length alone. CI, confidence interval.

{table(["Evaluation condition", "Training installation", "Test installation", "Macro-F1 (95% CI)"], S8, [22, 12, 12, 12])}

# Supplementary Figure S1. Nearest-neighbour similarity density

{figS1()}Density (kernel estimate recorded by the evaluation engine) of each trace's largest Jaccard index to another trace in its origin's training union, with peaks near 0.47 and 0.89 and a minimum near 0.53, the registered antimode. Vertical lines mark the registered antimode thresholds (0.528–0.535) and the analysed thresholds 0.7, 0.8 (primary) and 0.9. Drawn from the assembly record of the evaluation (assembly_info.json) by the released script figS1.py.

# Supplementary Equations

Equations S1–S9 support the learners, the de-duplication recall, the precision of ρ, sample-size planning and the secondary tests; Equation S10 is the pre-specified mechanism check reported in the Methodology section.

{supp_equations(eqmap, eqs)}

# Supplementary Note: pseudocode

{pseudo}

# Supplementary Note: registration record

**Kill criteria at the dry run.** K2 (both installations obtainable) and K3 (at least five shared families; 8, 9 and 8 found) were not triggered. K4 (label agreement) was triggered: two families were dropped and seven are unverified. K5 (interval precision) was triggered: the projected half-width was 5.8 points against 2, so the margin was raised to 3 points before scoring. K6 (compute) was triggered: a CPU plan with a sequence cap of 256 tokens was adopted.

**Amendments before the freeze.** (1) Ingestion releases each finished archive folder to bound memory. (2) Later stages stream records one at a time. (3) Label check against ClarAVy; revised dry run; CPU plan. (4) Dry-run training length and timing corrected. (5) Margin of 3 points; five seeds and 20 search configurations restored. Each amendment was tested against earlier outputs before installation.

**Deviation D1.** Decided on 8 November 2026 before any model was trained or scored: the primary near-duplicate threshold is the registered fallback 0.8 instead of the antimode (0.535, 0.532 and 0.528), which left 31–34 WinMET test traces per origin. The registered thresholds, 0.7 and 0.9 are reported as sensitivity analyses (Supplementary Table S5). Ledger entry 130,778, anchored by a trusted time-stamp at 01:39:20 UTC on 8 November 2026.

**Pre-specified checks.** P1 (zero attribution at masked positions) held in all six Transformer cells. P2 (completeness with 50 nodes) held in every cell. P3 (footprint overlaps less than behaviour) held before any model was trained. P4 (dry-run interval width) held (ratio 1.18). P5 (design effect at most 1.2) was not met. P6 (referred traces err more often) held only at WinMET origin 1. P7 (referral rate within its binomial interval) held everywhere. P8 (at most 10% borrowing by matched masks) was exceeded in one cell. P9 (monthly decline) was not supported.

# Supplementary Note: assembly

The analysed results file is recorded in the evidence ledger (entry 130,782, SHA-256 beginning b2e80138). Figures {{F:probes}}–{{F:transfer}} are drawn from that file by the released figure code (layout as in figures_final.py), and the build script generates Table {{T:hyp}} and fills Supplementary Table S4 from it; Supplementary Figure S1 is drawn from the assembly record of the same run (assembly_info.json); the build refuses any results file that is labelled as a synthetic rehearsal or whose digest differs from the ledger record.
"""
    return md
