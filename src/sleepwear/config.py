"""Project constants. No logic, no loader."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
EPOCHS_PATH = INTERIM / "epochs.parquet"
FEATURES_PATH = PROCESSED / "features.parquet"
MODELS_DIR = ROOT / "models"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
SUBJECTS_PATH = TABLES / "subjects.csv"  # metadata, never a feature

MOTION_FILE = "motion/{sid}_acceleration.txt"
HR_FILE = "heart_rate/{sid}_heartrate.txt"
LABEL_FILE = "labels/{sid}_labeled_sleep.txt"
STEPS_FILE = "steps/{sid}_steps.txt"  # reported for coverage only, not a feature

EPOCH_S = 30  # PSG scoring epoch, seconds
ACC_FS = 50  # resample rate for activity counts, Hz, as in the reference
COUNT_BAND_HZ = (3, 11)  # reference band-pass; undefined when the raw rate is <= 22 Hz
COUNT_SIGMA_S = 50  # Gaussian smoothing of activity counts
HR_MAX_GAP_S = 300  # 1 Hz HR interpolation bridges gaps of at most 5 min

# Raw PSG stage codes. 4 is R&K stage 4 (3 subjects only), not in the PhysioNet docs.
STAGE_CODES = {-1: "unscored", 0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "N4", 5: "REM"}
UNSCORED = -1
LABELED = (0, 1, 2, 3, 4, 5)
N_SUBJECTS = 31

INNER_FOLDS = 3  # grouped inner folds for tuning, training subjects only
SEEDS = (0, 1, 2)  # transformer seeds; results are the mean over seeds
