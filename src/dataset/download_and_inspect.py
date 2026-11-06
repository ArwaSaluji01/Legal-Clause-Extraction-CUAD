import hashlib
import json
import zipfile
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "cuad_data.zip"
EXTRACT_DIR = RAW_DIR / "cuad_release"

DATASET_URL = (
    "https://github.com/TheAtticusProject/"
    "cuad/raw/main/data.zip"
)

EXPECTED_SHA256 = (
    "f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a"
)


def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def download_dataset():

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists():

        existing_hash = calculate_sha256(ARCHIVE_PATH)

        if existing_hash == EXPECTED_SHA256:

            print("Verified dataset archive already exists.")
            return

        print("Existing archive failed checksum verification.")
        ARCHIVE_PATH.unlink()

    print("Downloading official CUAD dataset...")
    print(f"Source: {DATASET_URL}")

    response = requests.get(
        DATASET_URL,
        stream=True,
        timeout=(30, 300)
    )

    response.raise_for_status()

    with open(ARCHIVE_PATH, "wb") as file:

        for chunk in response.iter_content(chunk_size=1024 * 1024):

            if chunk:
                file.write(chunk)

    actual_hash = calculate_sha256(ARCHIVE_PATH)

    print("\nSHA256 verification:")
    print(f"Expected: {EXPECTED_SHA256}")
    print(f"Actual  : {actual_hash}")

    if actual_hash != EXPECTED_SHA256:

        ARCHIVE_PATH.unlink(missing_ok=True)

        raise ValueError(
            "Dataset checksum mismatch. "
            "The downloaded archive was not accepted."
        )

    print("\nDataset downloaded and checksum verified.")


def extract_dataset():

    print("\nExtracting dataset...")

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:

        for member in archive.infolist():

            target = (EXTRACT_DIR / member.filename).resolve()

            if not target.is_relative_to(EXTRACT_DIR.resolve()):

                raise ValueError(
                    f"Unsafe archive path: {member.filename}"
                )

        archive.extractall(EXTRACT_DIR)

    print("Extraction completed.")


def inspect_archive():

    print("\n" + "=" * 65)
    print("CUAD ARCHIVE FILE INSPECTION")
    print("=" * 65)

    for file_path in sorted(EXTRACT_DIR.rglob("*")):

        if file_path.is_file():

            relative_path = file_path.relative_to(EXTRACT_DIR)

            print(
                f"{relative_path} "
                f"({file_path.stat().st_size / 1024:.2f} KB)"
            )


def describe_structure(value, depth=0, max_depth=5):

    indent = "  " * depth

    if depth >= max_depth:

        print(f"{indent}... maximum inspection depth reached")
        return

    if isinstance(value, dict):

        print(f"{indent}Object with {len(value)} keys")

        for key, child in list(value.items())[:15]:

            print(f"{indent}{key}:")

            describe_structure(
                child,
                depth + 1,
                max_depth
            )

    elif isinstance(value, list):

        print(f"{indent}Array with {len(value)} elements")

        if value:

            print(f"{indent}First element:")

            describe_structure(
                value[0],
                depth + 1,
                max_depth
            )

    elif isinstance(value, str):

        preview = value[:100].replace("\n", " ")

        print(
            f"{indent}String | length={len(value)} "
            f"| preview={preview!r}"
        )

    elif isinstance(value, bool):

        print(f"{indent}Boolean: {value}")

    elif value is None:

        print(f"{indent}None")

    else:

        print(f"{indent}{type(value).__name__}: {value}")


def inspect_json_files():

    json_files = sorted(EXTRACT_DIR.rglob("*.json"))

    if not json_files:

        print("\nNo JSON files found in the archive.")
        return

    for json_path in json_files:

        print("\n" + "=" * 65)
        print(f"JSON FILE: {json_path.relative_to(EXTRACT_DIR)}")
        print("=" * 65)

        with open(json_path, "r", encoding="utf-8-sig") as file:

            data = json.load(file)

        describe_structure(data)


def main():

    download_dataset()

    # Create extraction directory if it does not exist.
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract only if the directory is empty.
    if not any(EXTRACT_DIR.iterdir()):
        extract_dataset()

    inspect_archive()

    inspect_json_files()


if __name__ == "__main__":
    main()