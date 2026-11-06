
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CHECKPOINT_DIR,
    DEVICE,
    RESULTS_DIR,
)

from src.evaluation.evaluate import predict_example


def main():
    parser = argparse.ArgumentParser(
        description="Extract a legal clause answer from a contract."
    )

    parser.add_argument(
        "--contract-file",
        type=str,
        required=True,
        help="Path to a plain-text contract file.",
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask about the contract.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "best_model"),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "prediction.json"),
    )

    args = parser.parse_args()

    contract_path = Path(args.contract_file)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Contract file not found: {contract_path}"
        )

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    if not args.question.strip():
        raise ValueError("Question cannot be empty.")

    context = contract_path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not context:
        raise ValueError("Contract file is empty.")

    print("=" * 60)
    print("LEGAL CLAUSE EXTRACTION")
    print("=" * 60)

    print(f"Contract: {contract_path.name}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint_path}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        use_fast=True,
    )

    print("Loading model...")
    model = AutoModelForQuestionAnswering.from_pretrained(
        checkpoint_path
    ).to(DEVICE)

    model.eval()

    example = {
        "question": args.question.strip(),
        "context": context,
    }

    print("\nExtracting answer...")

    result = predict_example(
        example=example,
        model=model,
        tokenizer=tokenizer,
        no_answer_threshold=args.threshold,
    )

    output = {
        "contract_file": str(contract_path),
        "question": args.question,
        "prediction": result["prediction"],
        "answer_score": result["answer_score"],
        "null_score": result["null_score"],
        "score_difference": result["score_difference"],
        "no_answer_threshold": args.threshold,
        "chunks_processed": result["chunks"],
        "device": str(DEVICE),
        "checkpoint": str(checkpoint_path),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("EXTRACTION RESULT")
    print("=" * 60)

    if result["prediction"]:
        print(f"Answer: {result['prediction']}")
    else:
        print("Answer: No answer predicted.")

    print(f"Chunks processed: {result['chunks']}")
    print(f"Answer score: {result['answer_score']}")
    print(f"Null score: {result['null_score']}")
    print(f"Score difference: {result['score_difference']}")

    print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()
