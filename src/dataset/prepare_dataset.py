import csv
import json
import re
import random
from collections import Counter
from pathlib import Path

import requests
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cuad_release"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_SOURCE = RAW_DIR / "train_separate_questions.json"
TEST_SOURCE = RAW_DIR / "test.json"

CATEGORY_SOURCE = (
    PROJECT_ROOT / "data" / "raw" / "category_descriptions.csv"
)

CATEGORY_URL = (
    "https://raw.githubusercontent.com/"
    "TheAtticusProject/cuad/main/category_descriptions.csv"
)

RANDOM_SEED = 42
VALIDATION_SIZE = 0.10


# --------------------------------------------------
# GENERAL UTILITIES
# --------------------------------------------------

def save_json(path, data):

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


def save_jsonl(path, records):

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:

        for record in records:

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required dataset file not found: {path}"
        )

    with open(path, "r", encoding="utf-8-sig") as file:

        return json.load(file)


def normalize_category(value):

    value = value.strip().lower()

    value = re.sub(r"\s+", " ", value)

    return value


# --------------------------------------------------
# CATEGORY DESCRIPTIONS
# --------------------------------------------------

def download_category_descriptions():

    CATEGORY_SOURCE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if CATEGORY_SOURCE.exists():

        print("Category descriptions already downloaded.")
        return

    print("Downloading category descriptions...")

    response = requests.get(
        CATEGORY_URL,
        timeout=60
    )

    response.raise_for_status()

    CATEGORY_SOURCE.write_bytes(response.content)

    print("Category descriptions downloaded.")


def load_categories():

    download_category_descriptions()

    categories = {}

    with open(
        CATEGORY_SOURCE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "Category (incl. context and answer)",
            "Description",
            "Answer Format",
            "Group"
        }

        if not required_columns.issubset(reader.fieldnames or []):

            raise ValueError(
                "Unexpected category CSV structure. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:

            category = row[
                "Category (incl. context and answer)"
            ].replace("Category: ", "").strip()

            description = row[
                "Description"
            ].replace("Description: ", "").strip()

            answer_format = row[
                "Answer Format"
            ].replace("Answer Format: ", "").strip()

            group = row[
                "Group"
            ].replace("Group: ", "").strip()

            if not category:
                continue

            categories[normalize_category(category)] = {
                "name": category,
                "description": description,
                "answer_format": answer_format,
                "group": group
            }

    if len(categories) != 41:

        raise ValueError(
            f"Expected 41 categories, found {len(categories)}."
        )

    print(f"Loaded {len(categories)} clause categories.")

    return categories


# --------------------------------------------------
# DATASET SCHEMA VALIDATION
# --------------------------------------------------

def validate_dataset_structure(dataset, filename):

    if not isinstance(dataset, dict):

        raise ValueError(
            f"{filename}: Root must be a dictionary."
        )

    if "data" not in dataset:

        raise ValueError(
            f"{filename}: Missing top-level 'data' field."
        )

    if not isinstance(dataset["data"], list):

        raise ValueError(
            f"{filename}: 'data' must be a list."
        )

    if not dataset["data"]:

        raise ValueError(
            f"{filename}: Dataset contains no contracts."
        )

    print(
        f"{filename}: "
        f"{len(dataset['data'])} contracts found."
    )


def extract_category(question, qa_id, categories):

    # First attempt: extract category from the question.
    match = re.search(
        r'related to\s+"([^"]+)"',
        question,
        flags=re.IGNORECASE
    )

    if match:

        category = match.group(1).strip()

        normalized = normalize_category(category)

        if normalized in categories:

            return categories[normalized]["name"]

    # Fallback: identify the category from the QA identifier.
    normalized_id = normalize_category(qa_id)

    for normalized, details in categories.items():

        category = normalize_category(details["name"])

        if (
            normalized_id.endswith("__" + category + "_0")
            or normalized_id.endswith("__" + category + "_1")
        ):

            return details["name"]

    raise ValueError(
        "Unable to identify category.\n"
        f"Question: {question}\n"
        f"QA ID: {qa_id}"
    )


# --------------------------------------------------
# CONTRACT EXTRACTION
# --------------------------------------------------

def extract_contracts(dataset, categories):

    contracts = []

    category_counter = Counter()

    total_questions = 0
    total_answer_spans = 0
    total_unanswerable = 0
    multiple_answer_questions = 0

    offset_errors = []

    for contract in dataset["data"]:

        title = contract["title"]

        paragraphs = contract["paragraphs"]

        if not paragraphs:

            raise ValueError(
                f"Contract has no paragraphs: {title}"
            )

        # CUAD's released files normally contain one
        # paragraph with the complete contract context.
        contexts = {
            paragraph["context"]
            for paragraph in paragraphs
        }

        if len(contexts) != 1:

            raise ValueError(
                f"Multiple distinct contexts found in {title}. "
                "Review this contract before processing."
            )

        context = next(iter(contexts))

        contract_id = title

        qa_records = []

        for paragraph in paragraphs:

            for qa in paragraph["qas"]:

                total_questions += 1

                question_id = qa["id"]

                question = qa["question"]

                category = extract_category(
                    question,
                    question_id,
                    categories
                )

                category_counter[category] += 1

                raw_answers = qa.get("answers", [])

                answers = []

                for answer in raw_answers:

                    answer_text = answer["text"]

                    answer_start = int(
                        answer["answer_start"]
                    )

                    answer_end = (
                        answer_start + len(answer_text)
                    )

                    # Validate original character offsets.
                    extracted_text = context[
                        answer_start:answer_end
                    ]

                    if extracted_text != answer_text:

                        offset_errors.append({
                            "contract": title,
                            "question_id": question_id,
                            "answer": answer_text,
                            "answer_start": answer_start,
                            "actual_text": extracted_text
                        })

                        continue

                    answers.append({
                        "text": answer_text,
                        "answer_start": answer_start,
                        "answer_end": answer_end
                    })

                if raw_answers and not answers:

                    raise ValueError(
                        "All answer offsets failed validation "
                        f"for {question_id}"
                    )

                if len(answers) > 1:

                    multiple_answer_questions += 1

                if not answers:

                    total_unanswerable += 1

                total_answer_spans += len(answers)

                qa_records.append({

                    "id": question_id,

                    "contract_id": contract_id,

                    "title": title,

                    "category": category,

                    "question": question,

                    "context": context,

                    "answers": answers,

                    "is_impossible": len(answers) == 0

                })

        contracts.append({

            "contract_id": contract_id,

            "title": title,

            "context": context,

            "qas": qa_records

        })

    if offset_errors:

        save_json(
            PROCESSED_DIR / "offset_errors.json",
            offset_errors
        )

        raise ValueError(
            f"{len(offset_errors)} answer offsets failed. "
            "See data/processed/offset_errors.json."
        )

    statistics = {

        "contract_count": len(contracts),

        "question_count": total_questions,

        "answer_span_count": total_answer_spans,

        "unanswerable_question_count": total_unanswerable,

        "multiple_answer_question_count": (
            multiple_answer_questions
        ),

        "category_count": len(category_counter),

        "category_question_counts": dict(
            category_counter
        )

    }

    return contracts, statistics


# --------------------------------------------------
# CREATE TRAINING EXAMPLES
# --------------------------------------------------

def create_training_examples(contracts):

    examples = []

    for contract in contracts:

        for qa in contract["qas"]:

            if qa["answers"]:

                # Each annotated span becomes a separate
                # training example.
                for answer_index, answer in enumerate(
                    qa["answers"]
                ):

                    examples.append({

                        "id": (
                            f"{qa['id']}_answer_{answer_index}"
                        ),

                        "contract_id": qa["contract_id"],

                        "title": qa["title"],

                        "category": qa["category"],

                        "question": qa["question"],

                        "context": qa["context"],

                        "answer_text": answer["text"],

                        "answer_start": answer["answer_start"],

                        "answer_end": answer["answer_end"],

                        "is_impossible": False

                    })

            else:

                # Preserve unanswerable questions.
                examples.append({

                    "id": qa["id"],

                    "contract_id": qa["contract_id"],

                    "title": qa["title"],

                    "category": qa["category"],

                    "question": qa["question"],

                    "context": qa["context"],

                    "answer_text": "",

                    "answer_start": -1,

                    "answer_end": -1,

                    "is_impossible": True

                })

    return examples


# --------------------------------------------------
# MAIN PREPARATION PIPELINE
# --------------------------------------------------

def main():

    random.seed(RANDOM_SEED)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 65)
    print("CUAD DATASET PREPARATION")
    print("=" * 65)

    categories = load_categories()

    train_data = load_json(TRAIN_SOURCE)

    test_data = load_json(TEST_SOURCE)

    validate_dataset_structure(
        train_data,
        "train_separate_questions.json"
    )

    validate_dataset_structure(
        test_data,
        "test.json"
    )

    print("\nProcessing training contracts...")

    train_contracts, train_stats = extract_contracts(
        train_data,
        categories
    )

    print("\nProcessing official test contracts...")

    test_contracts, test_stats = extract_contracts(
        test_data,
        categories
    )

    # Check for contract leakage between official splits.
    train_ids = {
        contract["contract_id"]
        for contract in train_contracts
    }

    test_ids = {
        contract["contract_id"]
        for contract in test_contracts
    }

    overlap = train_ids.intersection(test_ids)

    if overlap:

        raise ValueError(
            f"Contract leakage detected: {overlap}"
        )

    # Split training contracts into train and validation.
    train_ids, validation_ids = train_test_split(

        sorted(train_ids),

        test_size=VALIDATION_SIZE,

        random_state=RANDOM_SEED

    )

    train_ids = set(train_ids)

    validation_ids = set(validation_ids)

    def select_contracts(contracts, selected_ids):

        return [
            contract
            for contract in contracts
            if contract["contract_id"] in selected_ids
        ]

    final_train_contracts = select_contracts(
        train_contracts,
        train_ids
    )

    validation_contracts = select_contracts(
        train_contracts,
        validation_ids
    )

    # Explode answer spans only for training.
    train_examples = create_training_examples(
        final_train_contracts
    )

    # Keep all gold spans together for evaluation.
    validation_examples = [
        qa
        for contract in validation_contracts
        for qa in contract["qas"]
    ]

    test_examples = [
        qa
        for contract in test_contracts
        for qa in contract["qas"]
    ]

    # Save processed examples.
    save_jsonl(
        PROCESSED_DIR / "train.jsonl",
        train_examples
    )

    save_jsonl(
        PROCESSED_DIR / "validation.jsonl",
        validation_examples
    )

    save_jsonl(
        PROCESSED_DIR / "test.jsonl",
        test_examples
    )

    # Save original contract text separately for inference.
    all_contracts = (
        train_contracts + test_contracts
    )

    save_jsonl(
        PROCESSED_DIR / "contracts.jsonl",
        [
            {
                "contract_id": contract["contract_id"],
                "title": contract["title"],
                "context": contract["context"]
            }
            for contract in all_contracts
        ]
    )

    # Save category metadata.
    save_json(
        PROCESSED_DIR / "categories.json",
        list(categories.values())
    )

    # Save split manifest.
    split_manifest = {

        "random_seed": RANDOM_SEED,

        "validation_size": VALIDATION_SIZE,

        "train_contract_ids": sorted(train_ids),

        "validation_contract_ids": sorted(
            validation_ids
        ),

        "test_contract_ids": sorted(test_ids)

    }

    save_json(
        PROCESSED_DIR / "split_manifest.json",
        split_manifest
    )

    # Save statistics.
    statistics = {

        "training_source": train_stats,

        "test_source": test_stats,

        "final_train_contracts": len(train_ids),

        "validation_contracts": len(validation_ids),

        "test_contracts": len(test_ids),

        "final_train_examples": len(train_examples),

        "validation_examples": len(validation_examples),

        "test_examples": len(test_examples),

        "categories": len(categories)

    }

    save_json(
        PROCESSED_DIR / "dataset_statistics.json",
        statistics
    )

    print("\n" + "=" * 65)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 65)

    print(
        f"Categories             : {len(categories)}"
    )

    print(
        f"Training contracts     : {len(train_ids)}"
    )

    print(
        f"Validation contracts   : {len(validation_ids)}"
    )

    print(
        f"Official test contracts: {len(test_ids)}"
    )

    print(
        f"Training examples      : {len(train_examples)}"
    )

    print(
        f"Validation examples    : {len(validation_examples)}"
    )

    print(
        f"Test examples          : {len(test_examples)}"
    )

    print(
        f"Training answer spans  : "
        f"{sum(not x['is_impossible'] for x in train_examples)}"
    )

    print(
        f"Training no-answer     : "
        f"{sum(x['is_impossible'] for x in train_examples)}"
    )

    print("\nAll output files saved in:")
    print(PROCESSED_DIR)


if __name__ == "__main__":
    main()