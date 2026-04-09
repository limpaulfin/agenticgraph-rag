"""Evaluation metrics for QA: Exact Match, F1, Recall@k."""

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, remove articles/punctuation/whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 if normalized prediction == ground truth, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved_texts: list[str], gold_answer: str, gold_facts: list = None) -> float:
    """Check if gold answer appears in any retrieved chunk."""
    norm_gold = normalize_answer(gold_answer)
    for text in retrieved_texts:
        if norm_gold in normalize_answer(text):
            return 1.0
    return 0.0


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from list of {prediction, ground_truth, retrieved_texts}."""
    n = len(results)
    if n == 0:
        return {"em": 0.0, "f1": 0.0, "recall_at_k": 0.0, "n": 0}
    total_em = sum(exact_match(r["prediction"], r["ground_truth"]) for r in results)
    total_f1 = sum(f1_score(r["prediction"], r["ground_truth"]) for r in results)
    total_recall = sum(recall_at_k(r.get("retrieved_texts", []), r["ground_truth"]) for r in results)
    return {
        "em": total_em / n,
        "f1": total_f1 / n,
        "recall_at_k": total_recall / n,
        "n": n,
    }
