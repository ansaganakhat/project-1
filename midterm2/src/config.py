from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PAGE_URL = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
DATASET_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/"
    "smsspamcollection.zip"
)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RAW_ZIP_PATH = RAW_DATA_DIR / "smsspamcollection.zip"
RAW_DATA_PATH = RAW_DATA_DIR / "SMSSpamCollection"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "sms_spam_clean.csv"

RANDOM_STATE = 42
MAX_TOKENS = 10000
MAX_TFIDF_FEATURES = 5000
SEQUENCE_LENGTH = 80
BATCH_SIZE = 64
EPOCHS = 8
