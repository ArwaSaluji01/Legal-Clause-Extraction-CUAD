
import json
import torch

from config.config import (
    TRAIN_FILE,
    MODEL_MAX_LENGTH,
    DEVICE
)

from src.preprocessing.tokenizer import (
    CUADTokenizer
)

from src.model.qa_model import (
    LegalClauseQAModel
)


def load_test_examples(file_path):

    positive = None
    negative = None

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            example = json.loads(line)

            if example.get("is_impossible", False):

                if negative is None:
                    negative = example

            else:

                if positive is None:
                    positive = example

            if positive is not None and negative is not None:
                break

    if positive is None:
        raise ValueError(
            "No answerable training example found."
        )

    if negative is None:
        raise ValueError(
            "No unanswerable training example found."
        )

    return positive, negative


def validate_feature_alignment(
    example,
    features,
    tokenizer
):

    answer_text, answer_start, answer_end = (
        tokenizer._get_answer(example)
    )

    if answer_text is None:

        for feature in features:

            assert (
                feature["start_positions"]
                == feature["cls_index"]
            )

            assert (
                feature["end_positions"]
                == feature["cls_index"]
            )

        print("PASS: No-answer labels use CLS.")

        return

    aligned = False

    for feature in features:

        start = feature["start_positions"]
        end = feature["end_positions"]

        if start == feature["cls_index"]:
            continue

        start_offset = feature["offset_mapping"][start]
        end_offset = feature["offset_mapping"][end]

        if start_offset is None or end_offset is None:
            continue

        assert start <= end

        predicted_start = start_offset[0]
        predicted_end = end_offset[1]

        # The token span must cover the annotated
        # character span in the original contract.
        if (
            predicted_start <= answer_start
            and predicted_end >= answer_end
        ):

            aligned = True
            break

    if not aligned:

        raise AssertionError(
            "Answer was not aligned to any generated chunk."
        )

    print("PASS: Answer aligned to original contract offsets.")


def main():

    print("=" * 60)
    print("STAGE 3 TOKENIZATION VALIDATION")
    print("=" * 60)

    # 1. Initialize tokenizer
    tokenizer = CUADTokenizer()

    # 2. Load one positive and one negative example
    positive, negative = load_test_examples(
        TRAIN_FILE
    )

    print("\nSelected examples:")
    print(f"Positive: {positive['category']}")
    print(f"Negative: {negative['category']}")

    # 3. Tokenize positive example
    positive_features = tokenizer.tokenize_example(
        positive
    )

    print(
        f"\nPositive example chunks: "
        f"{len(positive_features)}"
    )

    validate_feature_alignment(
        positive,
        positive_features,
        tokenizer
    )

    # 4. Tokenize negative example
    negative_features = tokenizer.tokenize_example(
        negative
    )

    print(
        f"Negative example chunks: "
        f"{len(negative_features)}"
    )

    validate_feature_alignment(
        negative,
        negative_features,
        tokenizer
    )

    # 5. Verify maximum sequence length
    for feature in (
        positive_features + negative_features
    ):

        assert len(feature["input_ids"]) == MODEL_MAX_LENGTH

    print("PASS: Sequence length verified.")

    # 6. Initialize model
    print("\nLoading model for forward-pass test...")

    qa_model = LegalClauseQAModel(
        device=DEVICE
    )

    # 7. Run one forward pass
    feature = positive_features[0]

    input_ids = torch.tensor(
        [feature["input_ids"]],
        dtype=torch.long
    )

    attention_mask = torch.tensor(
        [feature["attention_mask"]],
        dtype=torch.long
    )

    outputs = qa_model.predict(
        input_ids,
        attention_mask
    )

    assert outputs.start_logits.shape == (
        1,
        MODEL_MAX_LENGTH
    )

    assert outputs.end_logits.shape == (
        1,
        MODEL_MAX_LENGTH
    )

    print("PASS: Model forward pass successful.")

    print(
        f"Start logits shape: {outputs.start_logits.shape}"
    )

    print(
        f"End logits shape: {outputs.end_logits.shape}"
    )

    print("\n" + "=" * 60)
    print("ALL STAGE 3 VALIDATION CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
