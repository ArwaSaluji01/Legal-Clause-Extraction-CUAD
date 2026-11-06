
import json
from pathlib import Path

from transformers import AutoTokenizer

from config.config import (
    MODEL_NAME,
    MODEL_MAX_LENGTH,
    DOCUMENT_STRIDE
)


class CUADTokenizer:

    def __init__(
        self,
        model_name=MODEL_NAME,
        max_length=MODEL_MAX_LENGTH,
        stride=DOCUMENT_STRIDE
    ):

        self.max_length = max_length
        self.stride = stride

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

        if not self.tokenizer.is_fast:
            raise ValueError(
                "A fast tokenizer is required for offset mappings."
            )

        self.cls_token_id = self.tokenizer.cls_token_id

        if self.cls_token_id is None:
            raise ValueError("Tokenizer CLS token is missing.")

    def _get_answer(self, example):

        answer_text = example.get("answer_text")
        answer_start = example.get("answer_start")
        answer_end = example.get("answer_end")

        # Training format
        if answer_text and answer_start is not None:

            answer_text = str(answer_text)
            answer_start = int(answer_start)

            if answer_end is None:
                answer_end = answer_start + len(answer_text)
            else:
                answer_end = int(answer_end)

            return answer_text, answer_start, answer_end

        # Alternative SQuAD-style annotation format
        answers = example.get("answers")

        if isinstance(answers, dict):

            texts = answers.get("text", [])
            starts = answers.get("answer_start", [])

            if texts and starts:

                answer_text = str(texts[0])
                answer_start = int(starts[0])

                return (
                    answer_text,
                    answer_start,
                    answer_start + len(answer_text)
                )

        elif isinstance(answers, list) and answers:

            first_answer = answers[0]

            if isinstance(first_answer, dict):

                answer_text = first_answer.get("text")
                answer_start = first_answer.get("answer_start")

                if answer_text and answer_start is not None:

                    answer_text = str(answer_text)
                    answer_start = int(answer_start)

                    return (
                        answer_text,
                        answer_start,
                        answer_start + len(answer_text)
                    )

        return None, None, None

    def _find_token_span(
        self,
        offsets,
        sequence_ids,
        answer_start,
        answer_end
    ):

        context_positions = [
            index
            for index, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1
        ]

        if not context_positions:
            return None

        start_token = None
        end_token = None

        for index in context_positions:

            token_start, token_end = offsets[index]

            if token_start == token_end:
                continue

            if (
                token_start <= answer_start
                and token_end > answer_start
            ):
                start_token = index

            if (
                token_start < answer_end
                and token_end >= answer_end
            ):
                end_token = index

        if start_token is None or end_token is None:
            return None

        if end_token < start_token:
            return None

        return start_token, end_token

    def tokenize_example(self, example):

        question = example["question"]
        context = example["context"]

        if not question or not context:
            raise ValueError(
                "Question and context cannot be empty."
            )

        answer_text, answer_start, answer_end = (
            self._get_answer(example)
        )

        if answer_text is not None:

            actual_text = context[
                answer_start:answer_end
            ]

            if actual_text != answer_text:

                raise ValueError(
                    f"Answer offset mismatch for {example.get('id')}"
                )

        encoded = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )

        features = []

        for feature_index in range(
            len(encoded["input_ids"])
        ):

            input_ids = encoded["input_ids"][feature_index]

            attention_mask = encoded[
                "attention_mask"
            ][feature_index]

            offsets = encoded[
                "offset_mapping"
            ][feature_index]

            sequence_ids = encoded.sequence_ids(
                feature_index
            )

            cls_index = input_ids.index(
                self.cls_token_id
            )

            start_position = cls_index
            end_position = cls_index

            if answer_text is not None:

                span = self._find_token_span(
                    offsets,
                    sequence_ids,
                    answer_start,
                    answer_end
                )

                if span is not None:

                    start_position, end_position = span

            # Preserve offsets only for context tokens.
            context_offsets = []

            for index, sequence_id in enumerate(
                sequence_ids
            ):

                if sequence_id == 1:
                    context_offsets.append(
                        list(offsets[index])
                    )
                else:
                    context_offsets.append(None)

            feature = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": start_position,
                "end_positions": end_position,
                "cls_index": cls_index,
                "offset_mapping": context_offsets,
                "example_id": example.get("id"),
                "contract_id": example.get("contract_id"),
                "category": example.get("category"),
                "feature_index": feature_index
            }

            features.append(feature)

        return features

    def tokenize_examples(self, examples):

        all_features = []

        for example in examples:

            features = self.tokenize_example(
                example
            )

            all_features.extend(features)

        return all_features


def load_jsonl_sample(file_path, limit=5):

    examples = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            examples.append(
                json.loads(line)
            )

            if len(examples) >= limit:
                break

    return examples


if __name__ == "__main__":

    from config.config import TRAIN_FILE

    print("Initializing CUAD tokenizer...")

    tokenizer = CUADTokenizer()

    examples = load_jsonl_sample(
        TRAIN_FILE,
        limit=3
    )

    features = tokenizer.tokenize_examples(
        examples
    )

    print("=" * 55)
    print("TOKENIZATION SUMMARY")
    print("=" * 55)

    print(f"Examples processed : {len(examples)}")
    print(f"Features generated : {len(features)}")

    for feature in features:

        print(
            f"Category: {feature['category']} | "
            f"Start: {feature['start_positions']} | "
            f"End: {feature['end_positions']}"
        )

    print("Tokenization completed.")
