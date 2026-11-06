
from pathlib import Path
import torch

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_FILE = PROCESSED_DATA_DIR / "train.jsonl"
VALIDATION_FILE = PROCESSED_DATA_DIR / "validation.jsonl"
TEST_FILE = PROCESSED_DATA_DIR / "test.jsonl"

MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "deepset/roberta-base-squad2"

MODEL_MAX_LENGTH = 384
DOCUMENT_STRIDE = 128

# --------------------------------------------------
# TRAINING CONFIGURATION
# --------------------------------------------------

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

NUM_EPOCHS = 3
WARMUP_RATIO = 0.1

GRADIENT_ACCUMULATION_STEPS = 8
MAX_GRAD_NORM = 1.0

# --------------------------------------------------
# TOKENIZATION
# --------------------------------------------------

PADDING = "max_length"
TRUNCATION = "only_second"

# --------------------------------------------------
# DEVICE CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_MIXED_PRECISION = torch.cuda.is_available()

# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

RANDOM_SEED = 42

# --------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------------------------

for directory in [
    MODEL_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    LOGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("=" * 55)
    print("CUAD PROJECT CONFIGURATION")
    print("=" * 55)

    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Model              : {MODEL_NAME}")
    print(f"Maximum length     : {MODEL_MAX_LENGTH}")
    print(f"Document stride    : {DOCUMENT_STRIDE}")
    print(f"Training batch     : {TRAIN_BATCH_SIZE}")
    print(f"Device             : {DEVICE}")
    print(f"Mixed precision    : {USE_MIXED_PRECISION}")

    print("=" * 55)
