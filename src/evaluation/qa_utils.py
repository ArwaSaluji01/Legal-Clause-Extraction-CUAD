
import re
import string


def get_gold_answers(example):
    """
    Extract all reference answers from supported CUAD JSONL schemas.
    Returns an empty list for unanswerable questions.
    """
    answers = []

    answer_text = example.get("answer_text")

    if isinstance(answer_text, str) and answer_text.strip():
        answers.append(answer_text.strip())

    raw_answers = example.get("answers")

    if isinstance(raw_answers, dict):
        texts = raw_answers.get("text", [])

        if isinstance(texts, str):
            texts = [texts]

        if isinstance(texts, list):
            answers.extend(
                text.strip()
                for text in texts
                if isinstance(text, str) and text.strip()
            )

    elif isinstance(raw_answers, list):
        for answer in raw_answers:
            if isinstance(answer, str) and answer.strip():
                answers.append(answer.strip())

            elif isinstance(answer, dict):
                text = answer.get("text")

                if isinstance(text, str) and text.strip():
                    answers.append(text.strip())

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(answers))


def normalize_answer(text):
    """
    Normalize text for Exact Match and token-level F1.
    """
    text = text.lower()
    text = "".join(
        character
        for character in text
        if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())

    return text


def exact_match_score(prediction, reference):
    return int(
        normalize_answer(prediction)
        == normalize_answer(reference)
    )


def token_f1_score(prediction, reference):
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()

    if not prediction_tokens or not reference_tokens:
        return int(prediction_tokens == reference_tokens)

    common_tokens = set(prediction_tokens) & set(reference_tokens)

    # Account for repeated words.
    from collections import Counter

    common_count = sum(
        (Counter(prediction_tokens) & Counter(reference_tokens)).values()
    )

    if common_count == 0:
        return 0.0

    precision = common_count / len(prediction_tokens)
    recall = common_count / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


def calculate_example_metrics(prediction, references):
    """
    Compare one prediction against all valid reference answers.

    For multiple references, use the best matching reference.
    """
    if not references:
        return {
            "exact_match": float(not prediction.strip()),
            "f1": float(not prediction.strip()),
        }

    exact_match = max(
        exact_match_score(prediction, reference)
        for reference in references
    )

    f1 = max(
        token_f1_score(prediction, reference)
        for reference in references
    )

    return {
        "exact_match": exact_match,
        "f1": f1,
    }
