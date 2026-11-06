
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    TEST_FILE,
    MODEL_MAX_LENGTH,
    DOCUMENT_STRIDE,
    DEVICE,
    CHECKPOINT_DIR,
    RESULTS_DIR,
)

from src.evaluation.qa_utils import (
    get_gold_answers,
    calculate_example_metrics,
)


def load_examples(file_path, limit=0):
    examples = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            examples.append(json.loads(line))

            if limit > 0 and len(examples) >= limit:
                break

    return examples


def create_inference_features(example, tokenizer):
    question = example["question"]
    context = example["context"]

    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=MODEL_MAX_LENGTH,
        stride=DOCUMENT_STRIDE,
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    features = []

    for feature_index in range(len(encoded["input_ids"])):
        sequence_ids = encoded.sequence_ids(feature_index)
        offsets = encoded["offset_mapping"][feature_index]

        context_offsets = [
            tuple(offset) if sequence_id == 1 else None
            for offset, sequence_id in zip(offsets, sequence_ids)
        ]

        features.append({
            "input_ids": encoded["input_ids"][feature_index],
            "attention_mask": encoded["attention_mask"][feature_index],
            "offset_mapping": context_offsets,
        })

    return features


def predict_example(
    example,
    model,
    tokenizer,
    no_answer_threshold=0.0,
    max_answer_tokens=64,
):
    model.eval()

    context = example["context"]

    features = create_inference_features(
        example,
        tokenizer,
    )

    best_answer_score = float("-inf")
    best_answer = ""
    minimum_null_score = float("inf")

    with torch.no_grad():
        for feature in features:
            input_ids = torch.tensor(
                [feature["input_ids"]],
                dtype=torch.long,
                device=DEVICE,
            )

            attention_mask = torch.tensor(
                [feature["attention_mask"]],
                dtype=torch.long,
                device=DEVICE,
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

            cls_positions = [
                index
                for index, token_id in enumerate(feature["input_ids"])
                if token_id == tokenizer.cls_token_id
            ]

            if not cls_positions:
                raise RuntimeError("CLS token not found in feature.")

            cls_position = cls_positions[0]

            null_score = (
                start_logits[cls_position].item()
                + end_logits[cls_position].item()
            )

            minimum_null_score = min(
                minimum_null_score,
                null_score,
            )

            offsets = feature["offset_mapping"]

            valid_positions = [
                index
                for index, offset in enumerate(offsets)
                if offset is not None and offset[1] > offset[0]
            ]

            if not valid_positions:
                continue

            top_k = min(20, len(valid_positions))

            valid_indices = torch.tensor(
                valid_positions,
                dtype=torch.long,
                device=DEVICE,
            )

            top_start_values, top_start_indices = torch.topk(
                start_logits[valid_indices],
                k=top_k,
            )

            top_end_values, top_end_indices = torch.topk(
                end_logits[valid_indices],
                k=top_k,
            )

            for start_value, start_local in zip(
                top_start_values.tolist(),
                top_start_indices.tolist(),
            ):
                start_index = valid_positions[start_local]

                for end_value, end_local in zip(
                    top_end_values.tolist(),
                    top_end_indices.tolist(),
                ):
                    end_index = valid_positions[end_local]

                    if end_index < start_index:
                        continue

                    if end_index - start_index + 1 > max_answer_tokens:
                        continue

                    char_start = offsets[start_index][0]
                    char_end = offsets[end_index][1]

                    if char_end <= char_start:
                        continue

                    answer = context[char_start:char_end].strip()

                    if not answer:
                        continue

                    score = start_value + end_value

                    if score > best_answer_score:
                        best_answer_score = score
                        best_answer = answer

    if best_answer_score == float("-inf"):
        return {
            "prediction": "",
            "answer_score": None,
            "null_score": minimum_null_score,
            "score_difference": None,
            "chunks": len(features),
        }

    score_difference = best_answer_score - minimum_null_score

    prediction = (
        ""
        if score_difference <= no_answer_threshold
        else best_answer
    )

    return {
        "prediction": prediction,
        "answer_score": best_answer_score,
        "null_score": minimum_null_score,
        "score_difference": score_difference,
        "chunks": len(features),
    }


def calculate_aggregate_metrics(predictions):
    if not predictions:
        raise ValueError("No predictions available for evaluation.")

    total = len(predictions)

    exact_match = sum(
        item["exact_match"] for item in predictions
    ) / total

    f1 = sum(
        item["f1"] for item in predictions
    ) / total

    return {
        "examples": total,
        "exact_match": round(exact_match * 100, 4),
        "f1": round(f1 * 100, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the CUAD extractive QA model."
    )

    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.0)

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "best_model"),
    )

    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("Limit cannot be negative.")

    checkpoint_dir = Path(args.checkpoint)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_dir}"
        )

    print("=" * 60)
    print("CUAD TEST SET EVALUATION")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Test example limit: {args.limit}")
    print(f"No-answer threshold: {args.threshold}")

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        use_fast=True,
    )

    print("Loading model...")

    model = AutoModelForQuestionAnswering.from_pretrained(
        checkpoint_dir
    ).to(DEVICE)

    model.eval()

    print("\nLoading test examples...")

    examples = load_examples(
        TEST_FILE,
        limit=args.limit,
    )

    print(f"Examples loaded: {len(examples)}")

    if not examples:
        raise RuntimeError("No test examples were loaded.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prediction_file = RESULTS_DIR / "test_predictions.jsonl"
    metrics_file = RESULTS_DIR / "test_metrics.json"

    evaluated = []
    category_results = defaultdict(list)

    with open(
        prediction_file,
        "w",
        encoding="utf-8",
    ) as output_file:

        for index, example in enumerate(examples, start=1):
            result = predict_example(
                example,
                model,
                tokenizer,
                no_answer_threshold=args.threshold,
            )

            gold_answers = get_gold_answers(example)

            scores = calculate_example_metrics(
                result["prediction"],
                gold_answers,
            )

            record = {
                "id": example.get("id"),
                "contract_id": example.get("contract_id"),
                "category": example.get("category"),
                "question": example["question"],
                "gold_answers": gold_answers,
                "prediction": result["prediction"],
                "answer_score": result["answer_score"],
                "null_score": result["null_score"],
                "score_difference": result["score_difference"],
                "chunks": result["chunks"],
                **scores,
            }

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            evaluated.append(record)

            category = example.get("category", "Unknown")
            category_results[category].append(record)

            print(
                f"[{index}/{len(examples)}] "
                f"Category: {category} | "
                f"Chunks: {result['chunks']} | "
                f"EM: {scores['exact_match']} | "
                f"F1: {scores['f1']:.4f}"
            )

    overall_metrics = calculate_aggregate_metrics(evaluated)

    per_category = {
        category: calculate_aggregate_metrics(records)
        for category, records in sorted(category_results.items())
    }

    final_metrics = {
        "evaluation_partition": "official_test",
        "checkpoint": str(checkpoint_dir),
        "device": str(DEVICE),
        "no_answer_threshold": args.threshold,
        "overall": overall_metrics,
        "per_category": per_category,
    }

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_metrics,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"Examples evaluated: {overall_metrics['examples']}")
    print(f"Exact Match: {overall_metrics['exact_match']:.4f}%")
    print(f"F1 Score: {overall_metrics['f1']:.4f}%")

    print(f"\nPredictions saved: {prediction_file}")
    print(f"Metrics saved: {metrics_file}")

    print("\nNOTE: These results reflect only the examples evaluated.")
    print("They are not full-test-set CUAD benchmark results.")


if __name__ == "__main__":
    main()
