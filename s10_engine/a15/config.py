"""A15 audit configuration: every constant fixed at S6-S8, plus the dataset specifications used at S9.

Nothing in this file may change after the freeze anchor (Algorithm 2); a change is a reported deviation.
Equation numbers refer to the S8 model; 'S7' and 'S6' refer to those stage reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

STUDY_ID = "A15-sandbox-artefact-audit/v1"          # h(0) of the ledger is SHA-256(enc(STUDY_ID)), eq. (25)

# ----------------------------------------------------------------------------------------------
# Decision constants (S6 hypotheses, S8 equations 16-20 and the freeze manifest)
# ----------------------------------------------------------------------------------------------
DELTA = 3.0                    # smallest effect of interest and equivalence margin, macro-F1 points (S6). Amendment 5
                               # (5 Nov 2026, before the freeze): 3, was 2, the K5 action registered at S6, because the
                               # corrected dry run projected a half-width of 5.8 points and the windows cannot widen
RHO_BOUND = 0.10               # H4 recovered-share bound (S6)
ALPHA_ONE_SIDED = 0.025        # one-sided level per claim; Holm with m_H = 4 (S8)
M_HOLM = 4
N_BOOT = 9_999                 # cluster-bootstrap replicates, eq. (16)-(17)
SEEDS = (11, 23, 37, 41, 53)   # n_s = 5 training seeds, averaged (S6-S8). Amendment 3 had kept the first three for the
                               # CPU plan; amendment 5 (5 Nov 2026) restores all five after the corrected K6 timing
N_SEARCH = 20                  # random-search configurations per learner (S7); amendment 3 had 8, amendment 5 restores 20
DROPOUT_GRID = (0.25, 0.5, 0.75, 1.0)   # artefact dropout p, eq. (3); p = 1 is deterministic F-masking
GAMMA_GRID = (0.0, 0.5, 1.0)  # class-weight exponent, eq. (7)
ETA_F = 0.05                   # referral budget for the base-model footprint share (decision D4)
IG_STEPS = (50, 100, 200)      # Gauss-Legendre nodes, raised if the completeness criterion fails (eq. 22)
IG_DELTA_MAX, IG_PASS_SHARE = 0.05, 0.95
K_MIN_TRAIN, K_MIN_TEST = 20, 10         # family thresholds per installation and origin (S6)
K3_MIN_FAMILIES = 5                      # kill criterion K3
K4_MIN_AGREEMENT = 0.80                  # kill criterion K4
K4_MIN_SHARED = 10                       # a family is judged against ClarAVy only with >= 10 shared SHA-256s in an
                                         # installation; below that it stays "unverified" (amendment 3, 4 Nov 2026)
K4_CLARAVY_MIN_CONF = 0.7                # ClarAVy labels used as the K4 reference need family_confidence >= 0.7
                                         # (about 90% accurate in the ClarAVy paper, section 6.2); low-confidence
                                         # generic names (e.g. heracles, jalapeno) are reported but not used to drop
K5_MAX_HALF_WIDTH = 2.0                  # kill criterion K5 (points)
K6_MAX_CPU_HOURS = 24.0                  # kill criterion K6
PAIRED_MIN_TRACES, PAIRED_MIN_FAMILIES = 200, 5   # same-binary paired test (S7 join rule)
N_REQ_RECALL, N_REQ_DEFF, N_REQ_CF = 0.80, 1.0, 1.0   # planning values for eq. (S6)

# ----------------------------------------------------------------------------------------------
# Sequence construction (eq. 1-2) and vocabulary (freeze manifest)
# ----------------------------------------------------------------------------------------------
BETA_MAX = 7                   # run-length bucket min(floor(log2 n_run), 7)
L_CAP = 256                    # CPU cap on the sequence length (K6); amendment 3 (CPU plan): 256, was 512
VOCAB_MIN_TRACES = 5           # token types seen in fewer training traces become <UNK>
PAD, MASK, UNK, OOS = "<PAD>", "<M>", "<UNK>", "<OOS>"
BETA_NEUTRAL = BETA_MAX + 1    # neutral run-length bucket used under the mask token, eq. (3)

# ----------------------------------------------------------------------------------------------
# Near-duplicates (eq. 8-9, S4) and matched masks (eq. 4, decision D5)
# ----------------------------------------------------------------------------------------------
TAU_J_FALLBACK = 0.8
TAU_J_KDE_RANGE = (0.5, 0.95)
LSH_CONFIG = {0.7: (32, 4), 0.8: (25, 5), 0.9: (16, 8)}   # tau_J -> (bands, rows), check C13
MASK_GRID_LADDER = ((10, 10), (5, 5), (4, 4), (3, 3))      # (frequency bands, position bins)
MASK_MAX_BORROW = 0.10

# ----------------------------------------------------------------------------------------------
# Splits (S7 split table): three rolling origins over 2025 for MELD-DS-448
# ----------------------------------------------------------------------------------------------
ORIGINS = (  # (origin id, validation month, first test day, last test day)
    (1, "2025-02", "2025-03-01", "2025-04-30"),
    (2, "2025-04", "2025-05-01", "2025-06-30"),
    (3, "2025-06", "2025-07-01", "2025-08-31"),
)
DRYRUN_PSEUDO_ORIGIN = ("2024-12", "2025-01-01", "2025-02-28")   # inside every origin's training period:
                                                                  # the dry run never opens a test window
DRYRUN_FRACTION = 1.0         # amendment 3 (4 Nov 2026): the dry run uses every pseudo-origin trace of the shared
                              # families; the planned 1% left 63 training traces and a model that predicted one family


@dataclass(frozen=True)
class Source:
    """One evidence source; 'files' are fnmatch patterns kept, 'skip' patterns are never downloaded."""
    code: str
    name: str
    licence: str
    host: str
    locator: str                      # HF repo id, Zenodo concept record id, git URL or Drive id
    revision: str = ""                # pinned commit or record version ('' = resolve and log the latest)
    files: tuple = ()
    skip: tuple = ()
    archive_password: str = ""        # documented in the data article (WinMET: 'infected')
    note: str = ""


SOURCES = {
    "I1": Source("I1", "MELD-DS-448", "CC BY-NC-SA 4.0", "huggingface", "MeldProject/MELD-DS-448",
                 revision="e2dbf22229067fbb12a746f50cf292bd6fe152a6",
                 files=("malicious_dataset_manifest.csv", "Dataset/cape_reports.7z.*", "Dataset/api_sequence.7z.*"),
                 skip=("Dataset/asm.7z.*", "Dataset/cape_reports_malicious_md.7z", "Dataset/api_sequence_split.7z.*",
                       ".DS_Store"),
                 note="CAPE JSON is the token source for both installations; API sequences are a cross-check"),
    "I2": Source("I2", "WinMET", "CC BY-NC 4.0", "zenodo", "12647555",
                 files=("WinMET_volume_*.7z", "*label*.json"),
                 archive_password="infected",
                 note="Raducu et al., Data Brief 63, 112273 (2025); MALVADA-processed CAPE JSON"),
    "I3": Source("I3", "Avast-CTU Public CAPEv2", "MIT (Licences.txt, commit b4210e5)", "gdrive",
                 "1ItUYKtr3hmjos8Hdd_e6rohuYSRgAAId",
                 note="Fallback only (K2); full reports, 13 GB; the reduced reports hold no API calls"),
    "MALVADA": Source("MALVADA", "MALVADA framework and 200 test reports", "GPL-3.0", "git",
                      "https://github.com/reverseame/malvada.git",
                      revision="24974027ef32e9a83aef07e42367bbffaa0ab9cf",
                      note="Curation filter (S7 harmonisation step 3) and real CAPE reports for the harness dry run"),
    "M": Source("M", "EMBER2024 Windows feature archives (family labels only)", "Apache-2.0", "huggingface",
                "joyce8/EMBER2024",
                files=("Win32_train.zip", "Win32_test.zip", "Win64_train.zip", "Win64_test.zip", "Dot_Net_train.zip",
                       "Dot_Net_test.zip"),
                note="Label check (K4) on shared SHA-256s only; never features. EMBER2024 publishes no labels-only "
                     "file, so the archives are read without unpacking and only ClarAVy labels of shared SHA-256s kept"),
}

# Report fields that carry labels or static data and must never reach a model (S7 pipeline step 1)
LABEL_BEARING_TOP = ("detections", "avclass_detection", "malfamily", "malfamily_tag", "malscore", "signatures",
                     "CAPE", "static", "strings", "virustotal", "suricata", "ttps", "mitre_attck",
                     "procmemory", "network", "dropped", "target")
ENV_FIELDS = ("cape_version", "guest_build", "hook_set_digest", "timeout", "route")   # gate fields, eq. (24)


@dataclass
class Paths:
    root: Path
    data: Path = field(init=False)
    work: Path = field(init=False)
    out: Path = field(init=False)
    ledger: Path = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root)
        self.data, self.work, self.out = self.root / "data", self.root / "work", self.root / "outputs"
        self.ledger = self.out / "ledger"
        for p in (self.data, self.work, self.out, self.ledger):
            p.mkdir(parents=True, exist_ok=True)
