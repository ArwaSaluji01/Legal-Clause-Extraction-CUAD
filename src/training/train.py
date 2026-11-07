import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

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
    MAX_GRAD_NORM,
    CHECKPOINT_DIR,
    LOGS_DIR,
    RANDOM_SEED,
)
from src.preprocessing.tokenizer import CUADTokenizer
from transformers import AutoModelForQuestionAnswering


# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# DATA SAMPLING
# ---------------------------------------------------------

def shuffled_jsonl_records(file_path, limit, seed, buffer_size=512):
    """
    Bounded-buffer shuffle.
    Does not load the complete JSONL dataset into memory.
    """

    rng = random.Random(seed)
    buffer = []
    yielded = 0

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            buffer.append(json.loads(line))

            if len(buffer) >= buffer_size:
                index = rng.randrange(len(buffer))
                yield buffer.pop(index)
                yielded += 1

                if yielded >= limit:
                    return

    rng.shuffle(buffer)

    for record in buffer:
        if yielded >= limit:
            break

        yield record
        yielded += 1


# ---------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------

def select_features_for_example(features, rng):
    """
    Answerable example:
      - one positive window containing the answer
      - at most one negative window

    Unanswerable example:
      - one randomly selected window

    An answerable example without a positive window is
    excluded and audited by the caller.
    """

    if not features:
        return [], "empty"

    positive = [
        feature
        for feature in features
        if (
            feature["start_positions"] != feature["cls_index"]
            or feature["end_positions"] != feature["cls_index"]
        )
    ]

    negative = [
        feature
        for feature in features
        if (
            feature["start_positions"] == feature["cls_index"]
            and feature["end_positions"] == feature["cls_index"]
        )
    ]

    is_answerable = (
        features[0].get("answer_text") is not None
        or any(
            feature["start_positions"] != feature["cls_index"]
            for feature in features
        )
    )

    if is_answerable:
        if not positive:
            return [], "unaligned_answer"

        selected = [rng.choice(positive)]

        if negative:
            selected.append(rng.choice(negative))

        return selected, "answerable"

    return [rng.choice(features)], "unanswerable"


# ---------------------------------------------------------
# CACHE GENERATION
# ---------------------------------------------------------

def build_feature_cache(
    file_path,
    cache_path,
    tokenizer,
    max_examples,
    seed,
    max_features,
):
    """
    Tokenizes a reproducible sample and saves only selected
    training windows. Full-document tokenization is retained
    so that answer locations are found correctly.
    """

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading existing feature cache: {cache_path}")

        cached = np.load(cache_path)

        arrays = {
            key: cached[key]
            for key in (
                "input_ids",
                "attention_mask",
                "start_positions",
                "end_positions",
            )
        }

        print("Cached features:", len(arrays["input_ids"]))
        return arrays

    rng = random.Random(seed)

    input_ids = []
    attention_masks = []
    start_positions = []
    end_positions = []

    audit = {
        "examples_seen": 0,
        "answerable_examples": 0,
        "unanswerable_examples": 0,
        "unaligned_answer_examples": [],
        "empty_examples": 0,
        "features_kept": 0,
        "categories": {},
    }

    print(f"\nBuilding feature cache: {file_path}")
    print(f"Examples to sample: {max_examples}")

    for example in shuffled_jsonl_records(
        file_path,
        limit=max_examples,
        seed=seed,
    ):
        audit["examples_seen"] += 1

        features = tokenizer.tokenize_example(example)

        # Attach answer information for safe classification.
        answer_text, _, _ = tokenizer._get_answer(example)

        for feature in features:
            feature["answer_text"] = answer_text

        selected, status = select_features_for_example(
            features,
            rng,
        )

        category = example.get("category", "UNKNOWN")
        audit["categories"][category] = (
            audit["categories"].get(category, 0) + 1
        )

        if status == "unaligned_answer":
            audit["unaligned_answer_examples"].append(
                example.get("id", "UNKNOWN")
            )
            continue

        if status == "empty":
            audit["empty_examples"] += 1
            continue

        if status == "answerable":
            audit["answerable_examples"] += 1
        else:
            audit["unanswerable_examples"] += 1

        for feature in selected:
            if len(input_ids) >= max_features:
                break

            input_ids.append(feature["input_ids"])
            attention_masks.append(feature["attention_mask"])
            start_positions.append(feature["start_positions"])
            end_positions.append(feature["end_positions"])

        if audit["examples_seen"] % 100 == 0:
            print(
                f"Examples processed: {audit['examples_seen']} | "
                f"Selected features: {len(input_ids)} | "
                f"Unaligned answers: "
                f"{len(audit['unaligned_answer_examples'])}"
            )

        if len(input_ids) >= max_features:
            break

    if not input_ids:
        raise RuntimeError("No usable features were generated.")

    arrays = {
        "input_ids": np.asarray(input_ids, dtype=np.uint16),
        "attention_mask": np.asarray(attention_masks, dtype=np.uint8),
        "start_positions": np.asarray(start_positions, dtype=np.uint16),
        "end_positions": np.asarray(end_positions, dtype=np.uint16),
    }

    np.savez_compressed(cache_path, **arrays)

    audit_path = cache_path.with_name(
        cache_path.stem + "_audit.json"
    )

    with open(audit_path, "w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)

    print("\nFeature cache completed.")
    print("Examples processed:", audit["examples_seen"])
    print("Answerable examples:", audit["answerable_examples"])
    print("Unanswerable examples:", audit["unanswerable_examples"])
    print("Unaligned answer examples:", len(audit["unaligned_answer_examples"]))
    print("Features retained:", len(input_ids))
    print("Categories represented:", len(audit["categories"]))
    print("Cache:", cache_path)
    print("Audit:", audit_path)

    return arrays


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

def make_dataloader(arrays, batch_size, shuffle, seed):
    dataset = TensorDataset(
        torch.from_numpy(arrays["input_ids"].astype(np.int64)),
        torch.from_numpy(arrays["attention_mask"].astype(np.int64)),
        torch.from_numpy(arrays["start_positions"].astype(np.int64)),
        torch.from_numpy(arrays["end_positions"].astype(np.int64)),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ---------------------------------------------------------
# CHECKPOINTS
# ---------------------------------------------------------

def save_checkpoint(
    model,
    tokenizer,
    output_dir,
    epoch,
    next_batch,
    global_step,
    validation_loss=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    tokenizer.tokenizer.save_pretrained(output_dir)

    metadata = {
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "validation_loss": validation_loss,
        "model_name": MODEL_NAME,
        "max_length": MODEL_MAX_LENGTH,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(
        output_dir / "training_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print(f"\nCheckpoint saved: {output_dir}")


def load_resume_checkpoint(model, resume_dir):
    resume_dir = Path(resume_dir)

    metadata_path = resume_dir / "training_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Resume metadata not found: {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    resumed_model = AutoModelForQuestionAnswering.from_pretrained(
        resume_dir
    ).to(DEVICE)

    print("Resuming from:", resume_dir)
    print("Resume metadata:", metadata)

    return (
        resumed_model,
        int(metadata["epoch"]),
        int(metadata["next_batch"]),
        int(metadata["global_step"]),
    )


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def evaluate_loss(model, dataloader, use_amp):
    model.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, starts, ends = [
                tensor.to(DEVICE, non_blocking=True)
                for tensor in batch
            ]

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=starts,
                    end_positions=ends,
                )

            batch_count = input_ids.size(0)
            total_loss += outputs.loss.item() * batch_count
            total_examples += batch_count

    if total_examples == 0:
        raise RuntimeError("Validation dataset is empty.")

    return total_loss / total_examples


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

def train(args):
    set_seed(RANDOM_SEED)

    use_amp = torch.cuda.is_available()

    print("=" * 65)
    print("CUAD TRAINING")
    print("=" * 65)
    print("Device:", DEVICE)
    print("GPU:", torch.cuda.get_device_name(0) if use_amp else "None")
    print("Model:", MODEL_NAME)
    print("Epochs:", args.epochs)
    print("Train examples:", args.train_examples)
    print("Validation examples:", args.validation_examples)
    print("Batch size:", args.batch_size)
    print("Gradient accumulation:", args.gradient_accumulation)
    print("Mixed precision:", use_amp)

    drive_root = Path("/content/drive/MyDrive/CUAD_Project")

    if drive_root.exists():
        cache_dir = drive_root / "cache"
    else:
        cache_dir = PROJECT_ROOT / "cache"

    cache_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    train_cache = cache_dir / f"train_{args.train_examples}_v1.npz"
    val_cache = cache_dir / f"validation_{args.validation_examples}_v1.npz"

    print("\nLoading tokenizer...")
    tokenizer = CUADTokenizer()

    train_arrays = build_feature_cache(
        TRAIN_FILE,
        train_cache,
        tokenizer,
        args.train_examples,
        RANDOM_SEED,
        args.max_train_features,
    )

    val_arrays = build_feature_cache(
        VALIDATION_FILE,
        val_cache,
        tokenizer,
        args.validation_examples,
        RANDOM_SEED + 1,
        args.max_validation_features,
    )

    train_loader = make_dataloader(
        train_arrays,
        args.batch_size,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_loader = make_dataloader(
        val_arrays,
        args.batch_size,
        shuffle=False,
        seed=RANDOM_SEED,
    )

    print("\nTraining features:", len(train_arrays["input_ids"]))
    print("Validation features:", len(val_arrays["input_ids"]))
    print("Training batches per epoch:", len(train_loader))

    if args.resume:
        model, start_epoch, resume_batch, global_step = (
            load_resume_checkpoint(
                None,
                args.resume,
            )
        )
    else:
        print("\nLoading pretrained QA model...")
        model = AutoModelForQuestionAnswering.from_pretrained(
            MODEL_NAME
        ).to(DEVICE)

        start_epoch = 1
        resume_batch = 0
        global_step = 0

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    latest_dir = CHECKPOINT_DIR / "latest_step"
    best_dir = CHECKPOINT_DIR / "best_model"

    best_validation_loss = float("inf")

    if args.resume:
        metadata_path = Path(args.resume) / "training_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as file:
            resume_metadata = json.load(file)

        previous_loss = resume_metadata.get("best_validation_loss")
        if previous_loss is not None:
            best_validation_loss = float(previous_loss)

    history_file = LOGS_DIR / "training_history.jsonl"

    for epoch in range(start_epoch, args.epochs + 1):
        print("\n" + "=" * 65)
        print(f"EPOCH {epoch}/{args.epochs}")
        print("=" * 65)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        seen_features = 0
        pending_gradients = 0

        epoch_loader = make_dataloader(
            train_arrays,
            args.batch_size,
            shuffle=True,
            seed=RANDOM_SEED + epoch,
        )

        skip_batches = resume_batch if epoch == start_epoch else 0

        if skip_batches:
            print("Skipping already completed batches:", skip_batches)

        for batch_index, batch in enumerate(epoch_loader):
            if batch_index < skip_batches:
                continue

            input_ids, attention_mask, starts, ends = [
                tensor.to(DEVICE, non_blocking=True)
                for tensor in batch
            ]

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=starts,
                    end_positions=ends,
                )
                loss = outputs.loss

            scaler.scale(
                loss / args.gradient_accumulation
            ).backward()

            batch_count = input_ids.size(0)
            total_loss += loss.item() * batch_count
            seen_features += batch_count
            pending_gradients += 1

            is_last_batch = (
                batch_index + 1 == len(epoch_loader)
            )

            if (
                pending_gradients == args.gradient_accumulation
                or is_last_batch
            ):
                # Correct the final partial accumulation group.
                if pending_gradients < args.gradient_accumulation:
                    correction = (
                        args.gradient_accumulation / pending_gradients
                    )

                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                pending_gradients = 0
                global_step += 1

                if global_step % 25 == 0:
                    average_loss = total_loss / max(seen_features, 1)

                    print(
                        f"Epoch {epoch} | "
                        f"Step {global_step} | "
                        f"Features {seen_features}/{len(train_arrays['input_ids'])} | "
                        f"Loss {average_loss:.4f}"
                    )

                if (
                    global_step % args.checkpoint_every == 0
                    and not is_last_batch
                ):
                    save_checkpoint(
                        model,
                        tokenizer,
                        latest_dir,
                        epoch,
                        batch_index + 1,
                        global_step,
                    )

                    # Store best loss in the latest checkpoint metadata.
                    metadata_path = latest_dir / "training_metadata.json"
                    with open(metadata_path, "r", encoding="utf-8") as file:
                        metadata = json.load(file)

                    metadata["best_validation_loss"] = best_validation_loss

                    with open(metadata_path, "w", encoding="utf-8") as file:
                        json.dump(metadata, file, indent=2)

        if seen_features == 0:
            raise RuntimeError(
                "No training features were processed in this epoch."
            )

        train_loss = total_loss / seen_features

        validation_loss = evaluate_loss(
            model,
            val_loader,
            use_amp,
        )

        print("\nEpoch summary")
        print("Training features:", seen_features)
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")

        record = {
            "epoch": epoch,
            "training_features": seen_features,
            "training_loss": train_loss,
            "validation_features": len(val_arrays["input_ids"]),
            "validation_loss": validation_loss,
            "global_step": global_step,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        with open(history_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        epoch_dir = CHECKPOINT_DIR / f"epoch_{epoch}"

        save_checkpoint(
            model,
            tokenizer,
            epoch_dir,
            epoch,
            0,
            global_step,
            validation_loss,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                model,
                tokenizer,
                best_dir,
                epoch,
                0,
                global_step,
                validation_loss,
            )

        # Update the emergency checkpoint to mark this epoch complete.
        save_checkpoint(
            model,
            tokenizer,
            latest_dir,
            epoch + 1,
            0,
            global_step,
            validation_loss,
        )

        metadata_path = latest_dir / "training_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        metadata["best_validation_loss"] = best_validation_loss

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        # The next epoch starts from its first batch.
        resume_batch = 0

    print("\n" + "=" * 65)
    print("TRAINING FINISHED")
    print("=" * 65)
    print("Best validation loss:", best_validation_loss)
    print("Best model:", best_dir)
    print("Training history:", history_file)
    print("Feature cache:", cache_dir)


# ---------------------------------------------------------
# COMMAND LINE
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-examples", type=int, default=4000)
    parser.add_argument("--validation-examples", type=int, default=400)
    parser.add_argument("--max-train-features", type=int, default=8000)
    parser.add_argument("--max-validation-features", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("Epochs must be at least 1.")

    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("Batch size and accumulation must be positive.")

    if args.checkpoint_every < 1:
        raise ValueError("Checkpoint interval must be positive.")

    drive_root = Path("/content/drive/MyDrive/CUAD_Project")
    cache_dir = (
        drive_root / "cache"
        if drive_root.exists()
        else PROJECT_ROOT / "cache"
    )

    if args.rebuild_cache:
        for cache_file in (
            cache_dir / f"train_{args.train_examples}_v1.npz",
            cache_dir / f"validation_{args.validation_examples}_v1.npz",
        ):
            if cache_file.exists():
                cache_file.unlink()
                print("Removed cache:", cache_file)

    train(args)


if __name__ == "__main__":
    main()
