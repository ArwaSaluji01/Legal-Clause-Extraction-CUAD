import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CHECKPOINT_DIR, DEVICE, MODEL_MAX_LENGTH


def main():
    checkpoint_dir = CHECKPOINT_DIR / "best_model"

    print("=" * 60)
    print("STAGE 4 CHECKPOINT VALIDATION")
    print("=" * 60)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_dir}"
        )

    metadata_path = checkpoint_dir / "training_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Training metadata not found: {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    print("Training metadata:")
    print(json.dumps(metadata, indent=4))

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        checkpoint_dir
    ).to(DEVICE)

    model.eval()

    sample = tokenizer(
        "What is the agreement?",
        "This is a sample agreement between two parties.",
        truncation="only_second",
        padding="max_length",
        max_length=MODEL_MAX_LENGTH,
        return_tensors="pt",
    )

    sample = {
        key: value.to(DEVICE)
        for key, value in sample.items()
    }

    with torch.no_grad():
        outputs = model(**sample)

    expected_shape = (1, MODEL_MAX_LENGTH)

    assert tuple(outputs.start_logits.shape) == expected_shape
    assert tuple(outputs.end_logits.shape) == expected_shape

    assert tokenizer.is_fast

    print("\nPASS: Checkpoint loaded successfully.")
    print("PASS: Fast tokenizer loaded successfully.")
    print("PASS: Start and end logits have expected shapes.")
    print(f"PASS: Model runs on {DEVICE}.")

    print("\nALL STAGE 4 VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
