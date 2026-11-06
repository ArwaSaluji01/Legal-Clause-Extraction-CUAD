import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                records.append(json.loads(line))

    return records


def validate_answer(context, answer):

    start = answer["answer_start"]
    text = answer["text"]

    if start < 0:
        raise ValueError("Negative answer offset found.")

    if context[start:start + len(text)] != text:

        raise ValueError(
            f"Answer alignment failed: {text[:100]}"
        )


def main():

    print("=" * 60)
    print("CUAD PROCESSED DATA VALIDATION")
    print("=" * 60)

    train = load_jsonl(DATA_DIR / "train.jsonl")

    validation = load_jsonl(
        DATA_DIR / "validation.jsonl"
    )

    test = load_jsonl(DATA_DIR / "test.jsonl")

    contracts = load_jsonl(
        DATA_DIR / "contracts.jsonl"
    )

    with open(
        DATA_DIR / "categories.json",
        "r",
        encoding="utf-8"
    ) as file:

        categories = json.load(file)

    with open(
        DATA_DIR / "split_manifest.json",
        "r",
        encoding="utf-8"
    ) as file:

        manifest = json.load(file)

    # Check category count.
    assert len(categories) == 41

    print("PASS: 41 categories loaded.")

    # Check contract counts.
    train_ids = set(manifest["train_contract_ids"])

    validation_ids = set(
        manifest["validation_contract_ids"]
    )

    test_ids = set(manifest["test_contract_ids"])

    assert len(train_ids) == 367
    assert len(validation_ids) == 41
    assert len(test_ids) == 102

    assert not train_ids.intersection(validation_ids)
    assert not train_ids.intersection(test_ids)
    assert not validation_ids.intersection(test_ids)

    assert len(contracts) == 510

    print("PASS: Contract counts and split isolation verified.")

    # Check training examples.
    for example in train:

        if example["is_impossible"]:

            assert example["answer_text"] == ""
            assert example["answer_start"] == -1

        else:

            start = example["answer_start"]

            answer = example["answer_text"]

            assert start >= 0

            assert (
                example["context"][
                    start:start + len(answer)
                ] == answer
            )

    print("PASS: Training answer spans aligned.")

    # Check validation and test examples.
    for dataset in [validation, test]:

        for example in dataset:

            answers = example["answers"]

            assert isinstance(answers, list)

            for answer in answers:

                validate_answer(
                    example["context"],
                    answer
                )

            assert (
                example["is_impossible"]
                == (len(answers) == 0)
            )

    print("PASS: Validation and test answer spans aligned.")

    # Verify contract isolation in examples.
    assert all(
        x["contract_id"] in train_ids
        for x in train
    )

    assert all(
        x["contract_id"] in validation_ids
        for x in validation
    )

    assert all(
        x["contract_id"] in test_ids
        for x in test
    )

    print("PASS: No document leakage.")

    # Verify all 41 categories occur in the data.
    category_names = {
        item["name"]
        for item in categories
    }

    observed_categories = {
        x["category"]
        for dataset in [train, validation, test]
        for x in dataset
    }

    assert observed_categories.issubset(category_names)

    print("PASS: Category names are valid.")

    print("\nDATASET SUMMARY")
    print("-" * 60)

    print(f"Contracts          : {len(contracts)}")
    print(f"Training examples  : {len(train)}")
    print(f"Validation examples: {len(validation)}")
    print(f"Test examples      : {len(test)}")
    print(f"Categories         : {len(categories)}")

    print("\nAll validation checks passed.")

    print("=" * 60)


if __name__ == "__main__":
    main()