"""Semantic evaluation metrics: BERTScore + LLM-as-Judge."""

import os
from pathlib import Path

_BERT_SCORER = None


def _get_scorer():
    """Lazy-load BERTScore scorer (avoids slow import at module level)."""
    global _BERT_SCORER
    if _BERT_SCORER is None:
        from bert_score import BERTScorer
        _BERT_SCORER = BERTScorer(lang="en", rescale_with_baseline=True)
    return _BERT_SCORER


def bert_score_f1(prediction: str, ground_truth: str) -> float:
    """BERTScore F1 between prediction and ground truth."""
    if not prediction or not ground_truth:
        return 0.0
    scorer = _get_scorer()
    P, R, F1 = scorer.score([prediction], [ground_truth])
    return F1.item()


def bert_score_batch(predictions: list[str], ground_truths: list[str]) -> list[float]:
    """BERTScore F1 for batch of predictions."""
    if not predictions:
        return []
    scorer = _get_scorer()
    P, R, F1 = scorer.score(predictions, ground_truths)
    return F1.tolist()


def llm_judge(prediction: str, ground_truth: str, question: str = "") -> float:
    """LLM-as-Judge: GPT-4o-mini judges semantic equivalence. Returns 1.0 or 0.0."""
    if not prediction or not ground_truth:
        return 0.0
    from m08_answer_generator import CLIENT
    if CLIENT is None:
        return 0.0
    prompt = (f"Are these two answers semantically equivalent for the question?\n"
              f"Question: {question}\n"
              f"Answer A: {prediction}\nAnswer B: {ground_truth}\n"
              f"Reply ONLY 'YES' or 'NO'.")
    resp = CLIENT.chat.completions.create(
        model="gpt-4o-mini", temperature=0.0, max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    return 1.0 if "YES" in resp.choices[0].message.content.strip().upper() else 0.0
