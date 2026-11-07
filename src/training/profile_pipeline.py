import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    TRAIN_FILE,
    MODEL_NAME,
    MODEL_MAX_LENGTH,
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
)
from src.preprocessing.tokenizer import CUADTokenizer
from src.model.qa_model import LegalClauseQAModel


def reservoir_sample(file_path, sample_size, seed):
    rng = random.Random(seed)
    sample = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if len(sample) < sample_size:
                sample.append(record)
            else:
                index = rng.randrange(0, len(sample) + 1)

                if index < sample_size:
                    sample[index] = record

    return sample


def benchmark_training_step(model, optimizer, feature, batch_size, device):
    input_ids = torch.tensor(
        [feature["input_ids"]] * batch_size,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.tensor(
        [feature["attention_mask"]] * batch_size,
        dtype=torch.long,
        device=device,
    )

    start_positions = torch.full(
        (batch_size,),
        int(feature["start_positions"]),
        dtype=torch.long,
        device=device,
    )

    end_positions = torch.full(
        (batch_size,),
        int(feature["end_positions"]),
        dtype=torch.long,
        device=device,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    def one_step():
        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        outputs.loss.backward()
        optimizer.step()

    # Warm-up
    for _ in range(3):
        one_step()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    iterations = 15
    start_time = time.perf_counter()

    for _ in range(iterations):
        one_step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    seconds_per_step = elapsed / iterations
    features_per_second = batch_size / seconds_per_step

    peak_memory_mb = None

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

    return {
        "batch_size": batch_size,
        "seconds_per_step": seconds_per_step,
        "features_per_second": features_per_second,
        "peak_memory_mb": peak_memory_mb,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile CUAD tokenization and model training speed."
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of training examples to sample.",
    )

    args = parser.parse_args()

    if args.sample_size < 10:
        raise ValueError("Use a sample size of at least 10.")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    print("=" * 65)
    print("CUAD PIPELINE PROFILING")
    print("=" * 65)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Maximum sequence length: {MODEL_MAX_LENGTH}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU total memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )
    else:
        print("WARNING: CUDA is unavailable. GPU benchmark will be skipped.")

    print("\nSampling training examples...")

    examples = reservoir_sample(
        TRAIN_FILE,
        sample_size=args.sample_size,
        seed=RANDOM_SEED,
    )

    if not examples:
        raise RuntimeError("No training examples were loaded.")

    print(f"Sampled examples: {len(examples)}")

    print("\nLoading tokenizer...")
    tokenizer = CUADTokenizer()

    print("Measuring tokenization...")

    all_features = []
    start_time = time.perf_counter()

    for example in examples:
        all_features.extend(
            tokenizer.tokenize_example(example)
        )

    tokenization_seconds = time.perf_counter() - start_time

    feature_count = len(all_features)
    average_features = feature_count / len(examples)

    tokenization_rate = feature_count / tokenization_seconds

    print("\nTOKENIZATION RESULTS")
    print("-" * 65)
    print(f"Examples sampled: {len(examples)}")
    print(f"Features generated: {feature_count}")
    print(f"Average features per example: {average_features:.2f}")
    print(f"Tokenization time: {tokenization_seconds:.2f} seconds")
    print(f"Tokenization rate: {tokenization_rate:.2f} features/second")

    print("\nLoading model for training-step benchmark...")

    qa_model = LegalClauseQAModel()
    model = qa_model.model

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("\nBENCHMARKING MODEL FORWARD/BACKWARD")
    print("-" * 65)

    benchmark_results = []

    if DEVICE.type == "cuda":
        batch_sizes = [1, 2]
    else:
        batch_sizes = [1]

    for batch_size in batch_sizes:
        try:
            result = benchmark_training_step(
                model=model,
                optimizer=optimizer,
                feature=all_features[0],
                batch_size=batch_size,
                device=DEVICE,
            )

            benchmark_results.append(result)

            print(f"\nBatch size: {batch_size}")
            print(
                f"Seconds per optimizer step: "
                f"{result['seconds_per_step']:.4f}"
            )
            print(
                f"Features per second: "
                f"{result['features_per_second']:.2f}"
            )

            if result["peak_memory_mb"] is not None:
                print(
                    f"Peak allocated GPU memory: "
                    f"{result['peak_memory_mb']:.2f} MB"
                )

        except torch.cuda.OutOfMemoryError:
            print(f"Batch size {batch_size}: GPU OUT OF MEMORY")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    if not benchmark_results:
        raise RuntimeError("No successful model benchmark completed.")

    best_result = max(
        benchmark_results,
        key=lambda item: item["features_per_second"],
    )

    estimated_total_features = (
        average_features * 20167
    )

    estimated_tokenization_seconds = (
        estimated_total_features / tokenization_rate
    )

    estimated_training_seconds = (
        estimated_total_features /
        best_result["features_per_second"]
    )

    print("\nROUGH FULL-DATASET ESTIMATE")
    print("-" * 65)
    print(
        f"Estimated training features: "
        f"{estimated_total_features:,.0f}"
    )
    print(
        f"Estimated tokenization time: "
        f"{estimated_tokenization_seconds / 3600:.2f} hours"
    )
    print(
        f"Estimated model training time using batch "
        f"{best_result['batch_size']}: "
        f"{estimated_training_seconds / 3600:.2f} hours per epoch"
    )
    print(
        "Note: estimates exclude validation, checkpoint saving, "
        "data-loading overhead, and performance variation."
    )

    print("\nProfiling completed. No project checkpoint was saved.")


if __name__ == "__main__":
    main()