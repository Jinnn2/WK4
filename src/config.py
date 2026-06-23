from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

ID_COL = "ID"
TARGET_COL = "cnt"
DATE_COL = "dteday"

SEED = 42
VALID_FRACTION = 0.2

