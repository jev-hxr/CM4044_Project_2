"""Central configuration: paths, task names, hyperparameters.

Every number reported in the project is produced with the values in this file
(unless a CLI flag overrides them, in which case the flag is logged next to
the result).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "tox21.csv.gz"
RAW_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
RAW_SHA256 = "45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360"
CURATED_PATH = DATA_DIR / "tox21_curated.csv"
CURATION_LOG = DATA_DIR / "curation_log.md"
CONFLICTS_PATH = DATA_DIR / "duplicate_conflicts.csv"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

# Pairs of Tox21 assays that probe the same receptor with different
# constructs / cell lines (full-length receptor vs. ligand-binding domain).
# A molecule that is active in one and inactive in the other is a
# "suspect" label: at least one of the two readouts is probably wrong
# (assay artefact, cytotoxicity, luciferase interference, ...).
RELATED_PAIRS = [("NR-AR", "NR-AR-LBD"), ("NR-ER", "NR-ER-LBD")]

SPLIT_FRACS = (0.8, 0.1, 0.1)  # train / valid / test, Bemis-Murcko scaffold split

# Fingerprints
ECFP_RADIUS = 2   # ECFP4
ECFP_BITS = 2048

# Noise grid (Task 2)
NOISE_KINDS = ["sym", "cc"]
NOISE_RATES = [0.05, 0.10, 0.20]
SEEDS = [0, 1, 2]

# Neural network hyperparameters (shared by every defence -> fair comparison)
MLP = dict(hidden=[1024, 256], dropout=0.3, lr=1e-3, weight_decay=1e-5,
           batch_size=128, max_epochs=80, patience=12)
GNN = dict(hidden=128, layers=4, dropout=0.2, lr=1e-3, weight_decay=1e-5,
           batch_size=128, max_epochs=80, patience=12)
RF = dict(n_estimators=100, min_samples_leaf=1, max_features="sqrt", n_jobs=-1)

# Defences (Task 3)
LABEL_SMOOTHING_EPS = 0.1
OOF_FOLDS = 5                        # cross-validation folds for out-of-fold probabilities
DETECTOR_MEMBERS = ["lr", "knn", "mlp"]  # diverse learners used by the noise detectors
DETECTOR_RF_TREES = 200
DETECTOR_MLP_EPOCHS = 25
ED_MIN_VOTES = 2                     # majority of 3 members must call the label wrong
