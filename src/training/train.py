
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    TRAIN_FILE,
    VALIDATION_FILE,
    MODEL_NAME,
    MODEL_MAX_LENGTH,
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_GRAD_NORM,
    CHECKPOINT_DIR,
    LOGS_DIR,
    RANDOM_SEED,
)
from src.preprocessing.tokenizer import CUADTokenizer
from src.model.qa_model import LegalClauseQAModel


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shuffled_jsonl_records(file_path, buffer_size=128, seed=42):
    """
    Shuffle records using a bounded buffer instead of loading
    the complete JSONL file into memory.
    """
    rng = random.Random(seed)
    buffer = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            buffer.append(json.loads(line))

            if len(buffer) >= buffer_size:
                index = rng.randrange(len(buffer))
                yield buffer.pop(index)

    rng.shuffle(buffer)

    while buffer:
        yield buffer.pop()


def generate_features(file_path, tokenizer, max_features=0, shuffle=False):
    """
    Tokenize JSONL examples lazily.

    max_features=0 means no feature limit.
    """
    if shuffle:
        records = shuffled_jsonl_records(
            file_path,
            seed=RANDOM_SEED,
        )
    else:
        def read_records():
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        yield json.loads(line)

        records = read_records()

    feature_count = 0

    for example in records:
        features = tokenizer.tokenize_example(example)

        for feature in features:
            yield feature
            feature_count += 1

            if max_features > 0 and feature_count >= max_features:
                return


def feature_to_tensors(feature):
    """
    Convert one tokenized feature into model input tensors.
    Offset mappings and metadata are not model inputs.
    """
    return {
        "input_ids": torch.tensor(
            [feature["input_ids"]],
            dtype=torch.long,
            device=DEVICE,
        ),
        "attention_mask": torch.tensor(
            [feature["attention_mask"]],
            dtype=torch.long,
            device=DEVICE,
        ),
        "start_positions": torch.tensor(
            [feature["start_positions"]],
            dtype=torch.long,
            device=DEVICE,
        ),
        "end_positions": torch.tensor(
            [feature["end_positions"]],
            dtype=torch.long,
            device=DEVICE,
        ),
    }


def save_checkpoint(model, tokenizer, output_dir, epoch, validation_loss):
    output_dir.mkdir(parents=True, exist_ok=True)

    model.model.save_pretrained(output_dir)
    tokenizer.tokenizer.save_pretrained(output_dir)

    metadata = {
        "epoch": epoch,
        "validation_loss": validation_loss,
        "model_name": MODEL_NAME,
        "max_length": MODEL_MAX_LENGTH,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Checkpoint saved: {output_dir}")


def train_one_epoch(
    model,
    optimizer,
    tokenizer,
    epoch,
    max_train_features,
    accumulation_steps,
):
    model.model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    feature_count = 0
    pending_gradients = 0

    features = generate_features(
        TRAIN_FILE,
        tokenizer,
        max_features=max_train_features,
        shuffle=True,
    )

    for feature in features:
        batch = feature_to_tensors(feature)

        outputs = model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"],
        )

        loss = outputs.loss
        (loss / accumulation_steps).backward()

        total_loss += loss.item()
        feature_count += 1
        pending_gradients += 1

        if pending_gradients == accumulation_steps:
            torch.nn.utils.clip_grad_norm_(
                model.model.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending_gradients = 0

        if feature_count % 10 == 0:
            print(
                f"Epoch {epoch} | "
                f"Training features: {feature_count} | "
                f"Average loss: {total_loss / feature_count:.4f}"
            )

    # Apply any remaining accumulated gradients.
    if pending_gradients > 0:
        correction = accumulation_steps / pending_gradients

        for parameter in model.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)

        torch.nn.utils.clip_grad_norm_(
            model.model.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if feature_count == 0:
        raise RuntimeError("No training features were generated.")

    return total_loss / feature_count, feature_count


def validate(
    model,
    tokenizer,
    max_validation_features,
):
    model.model.eval()

    total_loss = 0.0
    feature_count = 0

    with torch.no_grad():
        features = generate_features(
            VALIDATION_FILE,
            tokenizer,
            max_features=max_validation_features,
            shuffle=False,
        )

        for feature in features:
            batch = feature_to_tensors(feature)

            outputs = model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                start_positions=batch["start_positions"],
                end_positions=batch["end_positions"],
            )

            total_loss += outputs.loss.item()
            feature_count += 1

    if feature_count == 0:
        raise RuntimeError("No validation features were generated.")

    return total_loss / feature_count, feature_count


def main():
    parser = argparse.ArgumentParser(
        description="Train the CUAD extractive question-answering model."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-train-features",
        type=int,
        default=16,
        help="Maximum training features. Use 0 for the complete dataset.",
    )

    parser.add_argument(
        "--max-validation-features",
        type=int,
        default=8,
        help="Maximum validation features. Use 0 for the complete dataset.",
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("Epoch count must be at least 1.")

    if args.max_train_features < 0 or args.max_validation_features < 0:
        raise ValueError("Feature limits cannot be negative.")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Gradient accumulation must be at least 1.")

    set_seed(RANDOM_SEED)

    print("=" * 60)
    print("CUAD MODEL TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {args.epochs}")
    print(f"Training feature limit: {args.max_train_features}")
    print(f"Validation feature limit: {args.max_validation_features}")
    print(
        "Gradient accumulation:",
        args.gradient_accumulation_steps,
    )

    print("\nLoading tokenizer...")
    tokenizer = CUADTokenizer()

    print("\nLoading QA model...")
    model = LegalClauseQAModel()

    optimizer = AdamW(
        model.model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    run_dir = CHECKPOINT_DIR / "training_run"
    best_dir = CHECKPOINT_DIR / "best_model"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / "training_history.jsonl"
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 60)
        print(f"STARTING EPOCH {epoch}/{args.epochs}")
        print("=" * 60)

        train_loss, train_count = train_one_epoch(
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            epoch=epoch,
            max_train_features=args.max_train_features,
            accumulation_steps=args.gradient_accumulation_steps,
        )

        validation_loss, validation_count = validate(
            model=model,
            tokenizer=tokenizer,
            max_validation_features=args.max_validation_features,
        )

        print("\nEpoch results")
        print(f"Training features: {train_count}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation features: {validation_count}")
        print(f"Validation loss: {validation_loss:.4f}")

        record = {
            "epoch": epoch,
            "training_features": train_count,
            "training_loss": train_loss,
            "validation_features": validation_count,
            "validation_loss": validation_loss,
            "device": str(DEVICE),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        with open(log_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        save_checkpoint(
            model,
            tokenizer,
            run_dir / f"epoch_{epoch}",
            epoch,
            validation_loss,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                model,
                tokenizer,
                best_dir,
                epoch,
                validation_loss,
            )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Best checkpoint: {best_dir}")
    print(f"Training history: {log_file}")


if __name__ == "__main__":
    main()
